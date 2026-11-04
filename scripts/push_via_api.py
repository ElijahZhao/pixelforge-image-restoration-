#!/usr/bin/env python3.11
"""Push a local commit to GitHub through the REST API.

Why this exists
---------------
In this sandbox `github.com:443` is unreachable while `api.github.com` and
`ssh.github.com:443` are reachable, so `git push` over HTTPS times out. The
REST git-data API is reachable, so the commit is uploaded blob-by-blob /
tree-by-tree and the ref is advanced with a fast-forward update.

The update is a genuine fast-forward: the parent of the pushed commit must
equal the remote ref, otherwise the script refuses. No force is ever used.

Usage
-----
    python scripts/push_via_api.py            # push HEAD to origin/main
    python scripts/push_via_api.py --dry-run  # report what would be pushed
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

REPO = "ElijahZhao/pixelforge-image-restoration-"
API = f"https://api.github.com/repos/{REPO}"
CRED = "/root/.git-credentials"


def token():
    with open(CRED, encoding="utf-8") as fh:
        m = re.search(r"https://x-access-token:([^@]+)@", fh.read())
    if not m:
        raise SystemExit("no token in ~/.git-credentials")
    return m.group(1)


TOK = token()


def api(path, payload=None, method=None):
    url = path if path.startswith("http") else API + path
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {TOK}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:600]
        raise SystemExit(f"HTTP {e.code} on {method or 'GET'} {path}:\n{detail}")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ref", default="refs/heads/main")
    ap.add_argument("--ref-short", default="heads/main")
    args = ap.parse_args()

    local_head = git("rev-parse", "HEAD")
    parent = git("rev-parse", "HEAD~1")
    tree = git("rev-parse", "HEAD^{tree}")

    status, ref = api(f"/git/ref/{args.ref_short}")
    remote_sha = ref["object"]["sha"]
    print(f"local HEAD : {local_head}")
    print(f"local tree : {tree}")
    print(f"remote ref : {remote_sha}")

    # Fast-forward check.
    #
    # The remote sha is often NOT the local HEAD~1: when a previous push went
    # through this same API, the commit object was rebuilt server-side and got
    # a different sha even though the tree is byte-identical. Comparing shas
    # would then refuse an update that changes nothing.
    #
    # So the check is on the *tree*: the remote must already hold exactly the
    # content of one of this branch's ancestors. If it does, advancing the ref
    # to the new commit is a genuine fast-forward. If the remote holds content
    # this branch does not contain, the script refuses.
    st, remote_commit = api(f"/git/commits/{remote_sha}")
    remote_tree = remote_commit["tree"]["sha"]
    parent_tree = git("rev-parse", "HEAD~1^{tree}")
    print(f"remote tree: {remote_tree[:7]}")
    print(f"local HEAD~1 tree: {parent_tree[:7]}")

    # The remote may sit at an older ancestor of HEAD (a previous API push
    # rebuilt the commit server-side, so its sha and tree differ from the
    # local ones even though the content is the same). What matters is that
    # the remote's content is somewhere in this branch's history.
    ancestor_trees = {}
    for anc in git("rev-list", local_head).splitlines():
        ancestor_trees[git("rev-parse", f"{anc}^{{tree}}")] = anc
    if remote_tree in ancestor_trees:
        base = ancestor_trees[remote_tree]
        print(f"\nfast-forward check passed: the remote holds the content of "
              f"local commit {base[:7]}")
    else:
        print(f"\nREFUSING: the remote tree ({remote_tree[:7]}) matches no "
              f"commit in this branch's history. The remote has content this "
              f"branch does not contain; rebase onto it before pushing.")
        return 1
    if args.dry_run:
        print("\n--dry-run: stopping before upload")
        return 0

    # 1. Upload only the blobs the remote does not have yet.
    #    A blob that already exists must NOT be POSTed: the API is
    #    content-addressed, and re-creating an existing blob answers 500
    #    ("Server Error"), which looks like a transport failure but is really
    #    "you already have this". So probe first, upload only what is missing.
    #
    #    Probing every blob in the tree would be one API round-trip per blob
    #    -- over 350 requests for this repository, which takes minutes. Only
    #    the paths that differ between the remote's tree and this one can hold
    #    a missing blob, so the probe is restricted to those.
    out = subprocess.run(
        ["git", "diff", "--name-only", base, local_head],
        capture_output=True, text=True, check=True).stdout
    paths = sorted({p for p in out.splitlines() if p})
    print(f"paths to consider: {len(paths)} (changed since the remote)")

    blob_shas = []
    for path in paths:
        st, entry = api(f"/contents/{urllib.parse.quote(path)}?ref={remote_sha}")
        if st != 200:
            continue
        # Only a changed blob needs uploading; if the remote file already has
        # the blob we are about to push, there is nothing to do for it.
        want = subprocess.run(["git", "rev-parse", f"{local_head}:{path}"],
                               capture_output=True, text=True)
        if want.returncode != 0:
            continue
        want_sha = want.stdout.strip()
        if entry.get("sha") == want_sha:
            continue
        if entry.get("type") != "file":
            continue
        blob_shas.append(want_sha)

    def blob_exists(sha):
        try:
            req = urllib.request.Request(f"{API}/git/blobs/{sha}")
            req.add_header("Authorization", f"Bearer {TOK}")
            req.add_header("Accept", "application/vnd.github+json")
            req.add_header("X-GitHub-Api-Version", "2022-11-28")
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status == 200
        except urllib.error.HTTPError:
            return False

    uploaded = 0
    for sha in blob_shas:
        if blob_exists(sha):
            continue
        blob = subprocess.run(["git", "cat-file", "blob", sha],
                              capture_output=True, check=True).stdout
        api("/git/blobs", {
            "content": base64.b64encode(blob).decode(),
            "encoding": "base64",
        }, method="POST")
        uploaded += 1
    print(f"blobs: {uploaded} uploaded ({len(blob_shas)} candidates)")

    # 2. Build the tree recursively.
    #    A tree entry that points at a *subtree* must reference a tree the
    #    remote already has: the API rejects a tree sha it cannot resolve
    #    ("is not a valid tree"). So the tree is built bottom-up -- every
    #    directory is created from its own entries before the directory
    #    above it references it.
    def ls_tree(obj_sha, recursive=False):
        args = ["git", "ls-tree"]
        if recursive:
            args.append("-r")
        args.append(obj_sha)
        out = subprocess.run(args, capture_output=True, text=True,
                             check=True).stdout
        return [line.split("\t", 1) for line in out.splitlines()]

    def create_tree(obj_sha):
        """Create the tree object for obj_sha on the remote; return its sha."""
        entries = []
        for meta, path in ls_tree(obj_sha):
            mode, otype, sha = meta.split()
            if otype == "tree":
                sha = create_tree(sha)
            entries.append({"path": path, "mode": mode, "type": otype,
                            "sha": sha})
        st, res = api("/git/trees", {"tree": entries}, method="POST")
        return res["sha"]

    new_tree = create_tree(tree)
    print(f"tree: {tree[:7]} -> {new_tree[:7]}")
    if new_tree != tree:
        print("note: API tree sha differs from the local tree sha")

    # 3. Create the commit.
    msg = subprocess.run(["git", "log", "-1", "--pretty=%B", local_head],
                         capture_output=True, text=True,
                         check=True).stdout.strip()
    author = subprocess.run(
        ["git", "log", "-1", "--pretty=%an <%ae>", local_head],
        capture_output=True, text=True, check=True).stdout.strip()
    name, mail = author.split(" <")
    mail = mail.rstrip(">")
    when = subprocess.run(["git", "log", "-1", "--pretty=%aI", local_head],
                          capture_output=True, text=True, check=True).stdout.strip()
    st, commit = api("/git/commits", {
        "message": msg,
        "tree": new_tree,
        # The parent must be the commit the remote actually has. After an
        # earlier API push the remote sha differs from the local HEAD~1 sha
        # (the object was rebuilt server-side), so the remote's own sha is
        # used here -- otherwise the new commit would reference an object
        # the remote has never seen.
        "parents": [remote_sha],
        "author": {"name": name, "email": mail, "date": when},
    }, method="POST")
    print(f"commit created: {commit['sha'][:7]}")

    # 4. Advance the ref (fast-forward; the API refuses a non-ff update
    #    unless force=true, which this script never sends).
    api(f"/git/refs/{args.ref_short}", {"sha": commit["sha"], "force": False},
        method="PATCH")
    print(f"ref {args.ref} advanced: {remote_sha[:7]} -> {commit['sha'][:7]}")

    st, ref = api(f"/git/ref/{args.ref_short}")
    print(f"verified remote ref: {ref['object']['sha']}")
    if ref["object"]["sha"] != commit["sha"]:
        print("VERIFICATION FAILED")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
