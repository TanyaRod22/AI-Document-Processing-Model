import { useCallback, useRef, useState, type ReactNode } from "react";
import "./App.css";
import { askQuestion, deleteDocument, uploadDocument } from "./api";

const SUMMARY_PROMPT =
  "Write a concise summary of the document in 2–4 short paragraphs. Focus on the main purpose, key themes, and important facts. Use clear, plain language.";

const QUICK_PROMPTS: { id: string; label: string; prompt: string }[] = [
  {
    id: "summarize",
    label: "Summarize",
    prompt:
      "Provide a brief executive summary of the document in 2–3 short paragraphs.",
  },
  {
    id: "keypoints",
    label: "Key Points",
    prompt: "List the main key points as a clear bullet list.",
  },
  {
    id: "topics",
    label: "Find Topics",
    prompt: "What are the main topics or themes? List each with one short line of explanation.",
  },
  {
    id: "simple",
    label: "Explain Simply",
    prompt:
      "Explain the document in simple, non-technical language for someone unfamiliar with the subject.",
  },
];

type ChatMessage = { role: "user" | "assistant"; content: string };

const MAX_FILE_BYTES = 20 * 1024 * 1024;

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  return `${(n / 1024).toFixed(1)} KB`;
}

/** Tray with upward arrow — empty-state dropzone */
function IconUploadTray() {
  return (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M12 3v12m0 0l-4-4m4 4l4-4"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M4 14v5a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function IconDoc() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path d="M14 2v6h6" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M8 13h8M8 17h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function IconBolt() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" />
    </svg>
  );
}

function IconRobot() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
      <rect x="5" y="8" width="14" height="12" rx="2" stroke="currentColor" strokeWidth="1.5" />
      <path d="M9 8V6a3 3 0 0 1 6 0v2" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="9.5" cy="14" r="1" fill="currentColor" />
      <circle cx="14.5" cy="14" r="1" fill="currentColor" />
      <path d="M3 12h2M19 12h2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function IconSend() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M2 21l21-9L2 3v7l15 2-15 2v7z" />
    </svg>
  );
}

function IconSparkles() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M12 2l1.2 4.2L17 7l-3.8 1.2L12 12l-1.2-3.8L7 7l3.8-1.8L12 2zM5 14l.8 2.2L8 17l-2.2.8L5 20l-.8-2.2L2 17l2.2-.8L5 14z" />
    </svg>
  );
}

function IconList() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <circle cx="4" cy="6" r="1.5" />
      <circle cx="4" cy="12" r="1.5" />
      <circle cx="4" cy="18" r="1.5" />
      <path d="M8 6h12M8 12h12M8 18h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function IconSearch() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="10" cy="10" r="6" stroke="currentColor" strokeWidth="2" />
      <path d="M15 15l5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function IconBook() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <path
        d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  );
}

const ACTION_ICONS: Record<string, ReactNode> = {
  summarize: <IconSparkles />,
  keypoints: <IconList />,
  topics: <IconSearch />,
  simple: <IconBook />,
};

export default function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);
  const [fileSize, setFileSize] = useState<number | null>(null);
  const [summary, setSummary] = useState<string | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [chatBusy, setChatBusy] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const showError = useCallback((msg: string) => {
    setError(msg);
    window.setTimeout(() => setError(null), 6000);
  }, []);

  const resetLocal = useCallback(() => {
    setDocumentId(null);
    setFilename(null);
    setFileSize(null);
    setSummary(null);
    setSummaryLoading(false);
    setMessages([]);
    setInput("");
  }, []);

  const runAsk = useCallback(
    async (query: string): Promise<string> => {
      const { answer } = await askQuestion(query);
      return answer;
    },
    [],
  );

  const handleFile = useCallback(
    async (file: File | undefined) => {
      if (!file) return;
      if (file.size > MAX_FILE_BYTES) {
        showError("File is too large. Maximum size is 20MB.");
        return;
      }
      const lower = file.name.toLowerCase();
      if (!lower.endsWith(".pdf") && !lower.endsWith(".txt")) {
        showError("Please upload a PDF or TXT file.");
        return;
      }
      setUploadBusy(true);
      setError(null);
      try {
        if (documentId) {
          try {
            await deleteDocument(documentId);
          } catch {
            /* stale id or network; continue with new upload */
          }
        }
        const res = await uploadDocument(file);
        setDocumentId(res.document_id);
        setFilename(res.filename);
        setFileSize(file.size);
        setMessages([]);
        setSummary(null);
        setSummaryLoading(true);
        const answer = await runAsk(SUMMARY_PROMPT);
        setSummary(answer);
      } catch (e) {
        showError(e instanceof Error ? e.message : "Upload or summary failed.");
        resetLocal();
      } finally {
        setSummaryLoading(false);
        setUploadBusy(false);
      }
    },
    [documentId, resetLocal, runAsk, showError],
  );

  const onPickFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    void handleFile(f);
  };

  const onRemoveDocument = async () => {
    if (!documentId) {
      resetLocal();
      return;
    }
    try {
      await deleteDocument(documentId);
    } catch (e) {
      showError(e instanceof Error ? e.message : "Could not remove document.");
    } finally {
      resetLocal();
    }
  };

  const onSubmitChat = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = input.trim();
    if (!q || !documentId || chatBusy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setChatBusy(true);
    try {
      const answer = await runAsk(q);
      setMessages((m) => [...m, { role: "assistant", content: answer }]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: err instanceof Error ? err.message : "Something went wrong.",
        },
      ]);
    } finally {
      setChatBusy(false);
    }
  };

  const onQuickAction = async (label: string, prompt: string) => {
    if (!documentId || chatBusy || summaryLoading || uploadBusy) return;
    setMessages((m) => [...m, { role: "user", content: label }]);
    setChatBusy(true);
    try {
      const answer = await runAsk(prompt);
      setMessages((m) => [...m, { role: "assistant", content: answer }]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: err instanceof Error ? err.message : "Something went wrong.",
        },
      ]);
    } finally {
      setChatBusy(false);
    }
  };

  const hasUploadedDoc = Boolean(documentId && filename);
  const docReady = hasUploadedDoc && !uploadBusy;
  const actionsDisabled = !docReady || summaryLoading || chatBusy;
  const showLeftTools = hasUploadedDoc;
  const isEmptyLanding = !hasUploadedDoc && !uploadBusy;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="icon-well icon-well--md" aria-hidden>
            <IconDoc />
          </span>
          <div className="brand-text">
            <span className="brand-title">DocuMind</span>
            <span className="brand-tagline">AI Document Intelligence</span>
          </div>
        </div>
        <div className="powered">
          <span className="icon-well icon-well--sm" aria-hidden>
            <IconBolt />
          </span>
          Powered by RAG
        </div>
      </header>

      <input
        id="doc-upload"
        ref={fileInputRef}
        type="file"
        accept=".pdf,.txt,application/pdf,text/plain"
        className="visually-hidden"
        onChange={onPickFile}
        disabled={uploadBusy}
      />

      <div className="main">
        <aside className={`left-col ${isEmptyLanding ? "left-col--landing" : ""}`}>
          {!showLeftTools ? (
            <label
              htmlFor="doc-upload"
              className={`upload-zone upload-zone--hero ${dragOver ? "dragover" : ""} ${uploadBusy ? "upload-zone--busy" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                if (!uploadBusy) setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                if (!uploadBusy) void handleFile(e.dataTransfer.files?.[0]);
              }}
            >
              <span className="upload-zone__icon icon-well icon-well--lg" aria-hidden>
                <IconUploadTray />
              </span>
              <p className="upload-zone__title">Drop your document here</p>
              <p className="upload-zone__subtitle">PDF, TXT up to 20MB</p>
              {uploadBusy && <p className="upload-zone__status">Uploading…</p>}
            </label>
          ) : (
            <>
              <div className="file-chip">
                <span className="icon-well icon-well--md" aria-hidden>
                  <IconDoc />
                </span>
                <div className="meta">
                  <div className="name">{filename}</div>
                  <div className="size">{fileSize != null ? formatBytes(fileSize) : ""}</div>
                </div>
                <button
                  type="button"
                  className="icon-btn"
                  title="Remove document"
                  aria-label="Remove document"
                  onClick={() => void onRemoveDocument()}
                  disabled={uploadBusy || chatBusy}
                >
                  ×
                </button>
              </div>

              <button
                type="button"
                className="action-btn"
                style={{ marginTop: "-0.25rem" }}
                onClick={() => fileInputRef.current?.click()}
                disabled={uploadBusy || chatBusy}
              >
                Replace document
              </button>

              <div className="summary-card">
                <div className="card-head">
                  <span className="icon-well icon-well--md" aria-hidden>
                    <IconDoc />
                  </span>
                  <div>
                    <h2>Document Summary</h2>
                    {filename ? <div className="sub">{filename}</div> : null}
                  </div>
                </div>
                <div className="summary-body">
                  {summaryLoading && (
                    <>
                      <div className="skeleton w-90" />
                      <div className="skeleton w-80" />
                      <div className="skeleton w-70" />
                      <div className="skeleton w-90" />
                    </>
                  )}
                  {!summaryLoading && summary && (
                    <p style={{ margin: 0, whiteSpace: "pre-wrap" }}>{summary}</p>
                  )}
                </div>
              </div>

              <div className="quick-actions">
                <div className="divider-label">Quick actions</div>
                <div className="action-grid">
                  {QUICK_PROMPTS.map((a) => (
                    <button
                      key={a.id}
                      type="button"
                      className="action-btn"
                      disabled={actionsDisabled}
                      onClick={() => void onQuickAction(a.label, a.prompt)}
                    >
                      <span className="accent-icon">{ACTION_ICONS[a.id]}</span>
                      {a.label}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
        </aside>

        <section className="chat-panel">
          <div className="chat-head">
            <span className="icon-well icon-well--md" aria-hidden>
              <IconRobot />
            </span>
            <h2>Ask about your document</h2>
          </div>
          <div className="chat-scroll">
            {messages.length === 0 && (
              <div className="empty-chat">
                <span className="empty-chat__icon icon-well icon-well--md" aria-hidden>
                  <IconRobot />
                </span>
                {hasUploadedDoc ? (
                  <>Ask anything about your document — summaries, key points, or specific sections.</>
                ) : (
                  <>Upload a document to start asking questions</>
                )}
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`bubble ${msg.role}`}>
                {msg.content}
              </div>
            ))}
            {chatBusy && (
              <div className="bubble assistant" style={{ opacity: 0.85 }}>
                Thinking…
              </div>
            )}
          </div>
          <form className="chat-form" onSubmit={onSubmitChat}>
            <input
              type="text"
              placeholder={hasUploadedDoc ? "Ask a question…" : "Upload a document first…"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={!docReady || summaryLoading || chatBusy}
            />
            <button
              type="submit"
              className="send-btn"
              disabled={!docReady || summaryLoading || chatBusy || !input.trim()}
              aria-label="Send"
            >
              <IconSend />
            </button>
          </form>
        </section>
      </div>

      {error && <div className="toast" role="alert">{error}</div>}
    </div>
  );
}
