import { useStream } from "@langchain/langgraph-sdk/react";
import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

// The agent returns rich messages: `ai` content is an array of blocks
// (text / reasoning / function_call), `tool` messages carry a tool result,
// and `human` content is a plain string. We normalize each message into a
// simple shape the chat UI can render.
type ContentBlock =
  | { type: "text"; text: string }
  | { type: "reasoning"; [k: string]: unknown }
  | { type: "function_call"; name?: string; [k: string]: unknown }
  | { type: string; [k: string]: unknown };

type RawMessage = {
  id?: string;
  type: "human" | "ai" | "tool" | "system";
  content: string | ContentBlock[];
  name?: string;
  tool_calls?: { name?: string }[];
};

type ChatItem = {
  key: string;
  role: "user" | "assistant";
  text: string;
  toolCalls: string[];
};

const SUGGESTIONS = [
  "Summarize Apple's latest quarterly earnings.",
  "What did the most recent FOMC statement signal on rates?",
  "Draft a short investor note on this week's market-moving news.",
];

function textFromContent(content: string | ContentBlock[]): string {
  if (typeof content === "string") return content;
  return content
    .filter((b) => b.type === "text" && typeof (b as { text?: unknown }).text === "string")
    .map((b) => (b as { text: string }).text)
    .join("\n")
    .trim();
}

function toolCallsFromMessage(m: RawMessage): string[] {
  const fromField = (m.tool_calls ?? []).map((t) => t.name).filter(Boolean) as string[];
  const fromBlocks = Array.isArray(m.content)
    ? m.content
        .filter((b) => b.type === "function_call")
        .map((b) => (b as { name?: string }).name)
        .filter(Boolean)
    : [];
  return [...fromField, ...(fromBlocks as string[])];
}

// Collapse the raw message list into displayable chat turns:
// user bubbles, assistant text bubbles, and "used tool X" chips.
// Tool result messages and pure reasoning are intentionally hidden.
function toChatItems(messages: RawMessage[]): ChatItem[] {
  const items: ChatItem[] = [];
  messages.forEach((m, i) => {
    if (m.type === "human") {
      const text = textFromContent(m.content);
      if (text) items.push({ key: m.id ?? `u-${i}`, role: "user", text, toolCalls: [] });
    } else if (m.type === "ai") {
      const text = textFromContent(m.content);
      const toolCalls = toolCallsFromMessage(m);
      if (text || toolCalls.length) {
        items.push({ key: m.id ?? `a-${i}`, role: "assistant", text, toolCalls });
      }
    }
    // `tool` and `system` messages are omitted from the chat transcript.
  });
  return items;
}

function prettyToolName(name: string): string {
  const map: Record<string, string> = {
    task: "delegating to research agent",
    web_search: "searching the web",
    write_file: "drafting a note",
    edit_file: "editing a note",
    read_file: "reading a file",
    ls: "listing files",
  };
  return map[name] ?? name.replace(/_/g, " ");
}

export default function App() {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  const thread = useStream<{ messages: RawMessage[] }>({
    apiUrl: import.meta.env.VITE_LANGGRAPH_API_URL,
    assistantId: import.meta.env.VITE_LANGGRAPH_ASSISTANT_ID,
    messagesKey: "messages",
  });

  const items = useMemo(
    () => toChatItems((thread.messages as RawMessage[]) ?? []),
    [thread.messages],
  );

  // Auto-scroll to the newest message.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [items, thread.isLoading]);

  const submit = (value: string) => {
    const text = value.trim();
    if (!text || thread.isLoading) return;
    thread.submit({ messages: [{ type: "human", content: text }] });
    setInput("");
  };

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    submit(input);
  };

  const empty = items.length === 0;

  return (
    <div className="chat">
      <header className="chat-header">
        <div className="brand">
          <span className="brand-dot" />
          <div>
            <h1>Financial Research Desk</h1>
            <p className="subtitle">
              Earnings, rates &amp; market news — researched and sourced
            </p>
          </div>
        </div>
      </header>

      <div className="messages" ref={scrollRef}>
        {empty && (
          <div className="empty">
            <h2>What would you like researched?</h2>
            <p className="empty-sub">
              Ask about company earnings, macro data, or market-moving news.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="suggestion" onClick={() => submit(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {items.map((item) => (
          <div key={item.key} className={`row ${item.role}`}>
            {item.role === "assistant" && <div className="avatar">AI</div>}
            <div className="bubble-wrap">
              {item.toolCalls.length > 0 && (
                <div className="tools">
                  {item.toolCalls.map((name, i) => (
                    <span key={`${name}-${i}`} className="tool-chip">
                      <span className="spinner-dot" />
                      {prettyToolName(name)}
                    </span>
                  ))}
                </div>
              )}
              {item.text && (
                <div className={`bubble ${item.role}`}>
                  {item.text.split("\n").map((line, i) => (
                    <p key={i}>{line || "\u00a0"}</p>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {thread.isLoading && (
          <div className="row assistant">
            <div className="avatar">AI</div>
            <div className="bubble assistant typing">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </div>
          </div>
        )}
      </div>

      <form className="composer" onSubmit={onSubmit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about earnings, rates, or market news…"
          aria-label="Message"
        />
        {thread.isLoading ? (
          <button type="button" className="stop" onClick={() => thread.stop()}>
            Stop
          </button>
        ) : (
          <button type="submit" className="send" disabled={!input.trim()}>
            Send
          </button>
        )}
      </form>
    </div>
  );
}
