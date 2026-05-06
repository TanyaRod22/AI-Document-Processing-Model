function resolveApiBase(): string {
  const raw = import.meta.env.VITE_API_BASE;
  if (raw != null && String(raw).trim() !== "") {
    return String(raw).replace(/\/$/, "");
  }
  if (import.meta.env.DEV) {
    return "http://127.0.0.1:8003";
  }
  return "";
}

const API_BASE = resolveApiBase();
export const isApiBaseConfigured = Boolean(API_BASE);

function parseDetail(data: unknown): string {
  if (typeof data !== "object" || data === null || !("detail" in data)) {
    return "Request failed";
  }
  const d = (data as { detail: unknown }).detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    return d
      .map((x) => (typeof x === "object" && x !== null && "msg" in x ? String((x as { msg: unknown }).msg) : String(x)))
      .join("; ");
  }
  return String(d);
}

async function readError(res: Response): Promise<string> {
  try {
    const data: unknown = await res.json();
    return parseDetail(data);
  } catch {
    return res.statusText || "Request failed";
  }
}

function requireApiBase(): string {
  if (!API_BASE) {
    throw new Error(
      "API URL is not configured. Set VITE_API_BASE to your backend HTTPS URL before building (e.g. in Netlify environment variables).",
    );
  }
  return API_BASE;
}

export type UploadResult = {
  document_id: string;
  filename: string;
  chunks_created: number;
};

export async function uploadDocument(file: File): Promise<UploadResult> {
  const base = requireApiBase();
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${base}/upload`, { method: "POST", body });
  if (!res.ok) throw new Error(await readError(res));
  return res.json() as Promise<UploadResult>;
}

export type AskResult = { answer: string };

export async function askQuestion(query: string): Promise<AskResult> {
  const base = requireApiBase();
  const res = await fetch(`${base}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json() as Promise<AskResult>;
}

/** Removes vectors for this document from the index. Ignores 404 (already gone). */
export async function deleteDocument(documentId: string): Promise<void> {
  const base = requireApiBase();
  const res = await fetch(`${base}/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
  });
  if (res.status === 404) return;
  if (!res.ok) throw new Error(await readError(res));
}

export { API_BASE };
