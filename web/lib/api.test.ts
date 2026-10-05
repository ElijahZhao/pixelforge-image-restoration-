import { describe, it, expect, vi, afterEach } from "vitest";
import { predict } from "./api";

// `predict` is the only piece of the frontend with real branching logic
// (endpoint selection + error mapping), so it is the one worth testing
// without a DOM. Everything else in web/ is presentational.

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

function okResponse(body: unknown) {
  return {
    ok: true,
    status: 200,
    json: async () => body,
  } as unknown as Response;
}

describe("predict", () => {
  it("posts multipart form data to the same-origin /api path by default", async () => {
    const calls: { url: string; init: RequestInit }[] = [];
    vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return okResponse({ engine: "classical" });
    });

    const file = new File([new Uint8Array([1, 2, 3])], "a.png", {
      type: "image/png",
    });
    await predict(file, "sr", 4);

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("/api/predict");
    expect(calls[0].init.method).toBe("POST");
    const body = calls[0].init.body as FormData;
    expect(body.get("task")).toBe("sr");
    // scale must be stringified for multipart, not sent as a number.
    expect(body.get("scale")).toBe("4");
  });

  it("uses NEXT_PUBLIC_API_URL and strips a trailing slash", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://api.example.com/");
    let seen = "";
    vi.stubGlobal("fetch", async (url: string) => {
      seen = url;
      return okResponse({});
    });

    await predict(new File([], "a.png"), "lowlight", 2);

    expect(seen).toBe("https://api.example.com/api/predict");
  });

  it("throws with status and body text when the response is not ok", async () => {
    vi.stubGlobal("fetch", async () => ({
      ok: false,
      status: 429,
      text: async () => "too many requests; slow down",
    }) as unknown as Response);

    await expect(predict(new File([], "a.png"), "sr", 2)).rejects.toThrow(
      /429.*too many requests/
    );
  });
});
