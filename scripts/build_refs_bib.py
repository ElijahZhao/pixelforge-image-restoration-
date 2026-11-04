"""Build paper/refs.bib from the arXiv API, plus the three non-arXiv entries.

Why a script instead of hand-writing 27 entries: two arXiv ids written from
memory during the audit (KinD, RetinexNet) both resolved to unrelated papers.
Every id here is fetched, and the returned <title> is compared against the
expected title before the entry is emitted -- a mismatch aborts the build.

Venue strings (CVPR/ICCV/...) are not in the arXiv metadata, so they are declared
in ENTRIES below and only the author list / title / year are taken from the API.

    python scripts/build_refs_bib.py --out paper/refs.bib
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom"}

# key, arxiv id, expected title fragment (lower-case, checked against the API),
# venue, note
ENTRIES = [
    ("keles2021psnr", "2104.14868", "on the computation of psnr",
     "Picture Coding Symposium (PCS)", "2021"),
    ("srinivasan2026lol", "2610.38121", "what does lol measure",
     "arXiv preprint", "2026"),
    ("shah2026hyperbench", "2605.21671", "hyperbench",
     "arXiv preprint", "2026"),
    ("zhang2024seal", "2309.03020", "seal",
     "International Conference on Learning Representations (ICLR)", "2024"),
    ("su2025bridging", "2503.13074", "bridging the perception gap",
     "arXiv preprint", "2025"),
    ("parmar2022cleanfid", "2104.11222", "on aliased resizing",
     "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", "2022"),
    ("lucic2018gans", "1711.10337", "are gans created equal",
     "Advances in Neural Information Processing Systems (NeurIPS)", "2018"),
    ("musgrave2020reality", "2003.08505", "a metric learning reality check",
     "European Conference on Computer Vision (ECCV)", "2020"),
    ("dodge2019show", "1909.03004", "show your work",
     "Conference on Empirical Methods in Natural Language Processing (EMNLP)", "2019"),
    ("bouthillier2021variance", "2103.03098", "accounting for variance",
     "Conference on Machine Learning and Systems (MLSys)", "2021"),
    ("picard2021seed", "2109.08203", "torch.manual_seed(3407)",
     "arXiv preprint", "2021"),
    ("dong2014srcnn", "1501.00092", "image super-resolution using deep convolutional",
     "European Conference on Computer Vision (ECCV)", "2014"),
    ("ledig2017srgan", "1609.04802", "photo-realistic single image super-resolution",
     "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", "2017"),
    # arXiv comment: "To appear in CVPR 2017 workshop. Best paper award of the
    # NTIRE2017 workshop" -- so CVPRW, NOT the main conference.
    ("lim2017edsr", "1707.02921", "enhanced deep residual networks",
     "IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)", "2017"),
    ("zhang2018rcan", "1807.02758", "very deep residual channel attention",
     "European Conference on Computer Vision (ECCV)", "2018"),
    ("cai2023retinexformer", "2303.06705", "retinexformer",
     "IEEE/CVF International Conference on Computer Vision (ICCV)", "2023"),
    ("zhang2019kind", "1905.04161", "kindling the darkness",
     "ACM International Conference on Multimedia (ACM MM)", "2019"),
    ("wei2018retinex", "1808.04560", "deep retinex decomposition",
     "British Machine Vision Conference (BMVC)", "2018"),
    ("johnson2016perceptual", "1603.08155", "perceptual losses for real-time",
     "European Conference on Computer Vision (ECCV)", "2016"),
    ("simonyan2015vgg", "1409.1556", "very deep convolutional networks",
     "International Conference on Learning Representations (ICLR)", "2015"),
    ("blau2018perception", "1711.06077", "the perception-distortion tradeoff",
     "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", "2018"),
    # Crossref: the title containing "Fast" is the TPAMI 2019 extended version
    # (10.1109/tpami.2018.2865304); the CVPR 2017 paper is titled "Deep
    # Laplacian Pyramid Networks ..." (10.1109/cvpr.2017.618). arXiv 1710.01992
    # carries the TPAMI title, so cite it as the journal article, not CVPR.
    ("lai2019lapsrn", "1710.01992", "laplacian pyramid networks",
     "IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)", "2019"),
    ("kingma2015adam", "1412.6980", "adam: a method for stochastic optimization",
     "International Conference on Learning Representations (ICLR)", "2015"),
    ("he2016resnet", "1512.03385", "deep residual learning for image recognition",
     "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", "2016"),
]

# Non-arXiv entries, written out because they have no eprint to fetch.
STATIC = {
    "agustsson2017ntire": """@inproceedings{agustsson2017ntire,
  author    = {Agustsson, Eirikur and Timofte, Radu},
  title     = {{NTIRE} 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages     = {1122--1131},
  year      = {2017},
  doi       = {10.1109/CVPRW.2017.150}
}""",
    "wang2004ssim": """@article{wang2004ssim,
  author  = {Wang, Zhou and Bovik, Alan C. and Sheikh, Hamid R. and Simoncelli, Eero P.},
  title   = {Image Quality Assessment: From Error Visibility to Structural Similarity},
  journal = {IEEE Transactions on Image Processing},
  volume  = {13},
  number  = {4},
  pages   = {600--612},
  year    = {2004},
  doi     = {10.1109/TIP.2003.819861}
}""",
    "cawley2010overfitting": """@article{cawley2010overfitting,
  author  = {Cawley, Gavin C. and Talbot, Nicola L. C.},
  title   = {On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation},
  journal = {Journal of Machine Learning Research},
  volume  = {11},
  pages   = {2079--2107},
  year    = {2010}
}""",
}

# The conference version of a paper is not always the arXiv version, and the
# author list is where that shows up most often. When the venue string names a
# conference, the proceedings' list is authoritative -- so any mismatch is
# recorded here rather than silently inheriting the preprint's.
#
# 2103.03098: MLSys 2021 (Proceedings of Machine Learning and Systems 3) lists
#  16 authors; arXiv v1 lists 17 (it adds Dmitriy Serdyuk, between Michalski
#  and Arbel). Verified by fetching the MLSys 2021 proceedings index, not from
#  memory. The 7th author is "Naz Sepah" on arXiv and "Nazanin Mohammadi
#  Sepahvand" in the proceedings -- same person, full name.
AUTHOR_OVERRIDE: dict[str, list[str]] = {
    "2103.03098": [
        "Xavier Bouthillier", "Pierre Delaunay", "Mirko Bronzi",
        "Assya Trofimov", "Brennan Nichyporuk", "Justin Szeto",
        "Nazanin Mohammadi Sepahvand", "Edward Raff", "Kanika Madan",
        "Vikram Voleti", "Samira Ebrahimi Kahou", "Vincent Michalski",
        "Tal Arbel", "Chris Pal", "Gaël Varoquaux", "Pascal Vincent",
    ],
}

# Titles with acronyms or mixed case that BibTeX would otherwise lowercase.
PROTECT = [
    "SEAL", "HyperBench", "NTIRE", "LOL", "GAN", "GANs", "PSNR", "CNN", "RGB",
    "YCbCr", "SR", "torch.manual_seed", "Adam",
]


def fetch(ids: list[str]) -> dict[str, tuple[str, list[str], str, str]]:
    """id -> (title, authors, published year, primary category).

    The primary category is read from the API rather than assumed: writing
    "cs.CV" for every preprint is wrong for HyperBench (2605.21671), whose
    primary category is eess.IV (image and video processing) with cs.CV
    only as a secondary category.

    The request is retried because arXiv throttles bursts of API calls and
    a CI job that shares an IP with everyone else's can be handed a 429 or
    a 503. A transient refusal must not be recorded as "the bibliography
    changed": that would make the check fail for a reason that has nothing
    to do with the manuscript. Only a response that actually arrives is
    compared, and a missing entry is still a hard error below.
    """
    url = ("https://export.arxiv.org/api/query?id_list=" + ",".join(ids)
           + "&max_results=" + str(len(ids)))
    last: Exception | None = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "pixelforge-refs-builder/1.0 "
                              "(preprint submission bibliography check)",
            })
            with urllib.request.urlopen(req, timeout=60) as fh:
                root = ET.fromstring(fh.read())
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (attempt + 1))
    else:
        # Exit 2 means "could not run", distinct from exit 1's "the
        # bibliography is wrong". The CI step reads the difference: an
        # arXiv outage is not a defect in the manuscript.
        print(f"arXiv API unreachable after 5 attempts: {last}", file=sys.stderr)
        sys.exit(2)

    out = {}
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        title = " ".join(e.find("a:title", NS).text.split())
        authors = [" ".join(a.find("a:name", NS).text.split())
                   for a in e.findall("a:author", NS)]
        year = e.find("a:published", NS).text[:4]
        pc = e.find("arxiv:primary_category", NS)
        primary = pc.get("term") if pc is not None else "cs.CV"
        out[aid] = (title, authors, year, primary)
    return out


def bib_author(names: list[str]) -> str:
    """"Chao Dong" -> "Dong, Chao"; joins with ' and '."""
    out = []
    for n in names:
        parts = n.split()
        if len(parts) == 1:
            out.append(parts[0])
        else:
            out.append(parts[-1] + ", " + " ".join(parts[:-1]))
    return " and ".join(out)


def brace_title(t: str) -> str:
    """Render an arXiv title so that it survives a LaTeX/BibTeX round trip.

    Two separate jobs, and conflating them is what broke the Picard entry:
    the token list below protects capitalisation (``{GAN}`` keeps the
    capitals from being lowercased by the bibliography style), while the
    escaping handles characters that are *syntactic* in LaTeX. The title
    "torch.manual_seed(3407) is all you need" contains an underscore and
    a pair of parentheses; in a ``plain`` bibliography style these are
    interpreted as a subscript and a group, and the entry fails to compile
    with "Extra }, or forgotten \\endgroup".

    Escaping happens first: an escaped character must not then be wrapped
    in a protection brace.
    """
    t = t.replace("\\", r"\textbackslash{}")
    for src, dst in (("_", r"\_"), ("&", r"\&"), ("%", r"\%"),
                     ("$", r"\$"), ("#", r"\#")):
        t = t.replace(src, dst)
    # Parentheses are only special inside the protected token, which is
    # where they actually appear; escape them there rather than everywhere,
    # so ordinary titles keep their plain parentheses.
    t = re.sub(r"(?<![\w{])torch\.manual_seed\((\d+)\)",
               r"torch.manual\_seed(\1)", t)
    for tok in PROTECT:
        t = re.sub(rf"(?<![\w{{]){re.escape(tok)}(?![\w}}])", "{" + tok + "}", t)
    return "{" + t + "}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/refs.bib")
    args = ap.parse_args()

    ids = [e[1] for e in ENTRIES]
    meta = fetch(ids)
    missing = [i for i in ids if i not in meta]
    if missing:
        raise SystemExit(f"arXiv returned nothing for: {missing}")

    chunks = ["% Generated by scripts/build_refs_bib.py -- do not hand-edit.",
              "% Every arXiv id is fetched and its title checked before writing.",
              ""]
    for key, aid, expect, venue, year in ENTRIES:
        title, authors, _, primary = meta[aid]
        authors = AUTHOR_OVERRIDE.get(aid, authors)
        if expect.lower() not in title.lower():
            raise SystemExit(
                f"title mismatch for {aid}\n  expected to contain: {expect}\n  got: {title}")
        if venue == "arXiv preprint":
            entry = (f"@misc{{{key},\n"
                     f"  author       = {{{bib_author(authors)}}},\n"
                     f"  title        = {brace_title(title)},\n"
                     f"  year         = {{{year}}},\n"
                     f"  eprint       = {{{aid}}},\n"
                     f"  archivePrefix= {{arXiv}},\n"
                     f"  primaryClass = {{{primary}}},\n"
                     f"  note         = {{arXiv preprint}},\n"
                     f"}}")
        elif "Transactions" in venue or "Journal" in venue:
            # Journal version: @article + journal, not @inproceedings.
            # Some arXiv entries carry the extended-journal title -- LapSRN
            # 1710.01992 is "Fast and Accurate ..." (TPAMI 2019), while the
            # CVPR 2017 paper is "Deep Laplacian Pyramid Networks ...".
            entry = (f"@article{{{key},\n"
                     f"  author       = {{{bib_author(authors)}}},\n"
                     f"  title        = {brace_title(title)},\n"
                     f"  journal      = {{{venue}}},\n"
                     f"  year         = {{{year}}},\n"
                     f"  eprint       = {{{aid}}},\n"
                     f"  archivePrefix= {{arXiv}},\n"
                     f"}}")
        else:
            entry = (f"@inproceedings{{{key},\n"
                     f"  author       = {{{bib_author(authors)}}},\n"
                     f"  title        = {brace_title(title)},\n"
                     f"  booktitle    = {{{venue}}},\n"
                     f"  year         = {{{year}}},\n"
                     f"  eprint       = {{{aid}}},\n"
                     f"  archivePrefix= {{arXiv}},\n"
                     f"}}")
        chunks.append(entry)
    chunks.extend(STATIC.values())

    with open(args.out, "w") as fh:
        fh.write("\n\n".join(chunks) + "\n")
    print(f"wrote {args.out}: {len(ENTRIES) + len(STATIC)} entries")


if __name__ == "__main__":
    main()
