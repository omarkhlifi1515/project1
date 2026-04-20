import { useMemo, useState } from "react";
import { adminSync, chatQuery, login, register } from "./api";

export function App() {
  const [token, setToken] = useState(localStorage.getItem("token") || "");
  const [role, setRole] = useState(localStorage.getItem("role") || "");
  const [email, setEmail] = useState(localStorage.getItem("email") || "");
  const [authMode, setAuthMode] = useState("login");
  const [authForm, setAuthForm] = useState({
    email: "",
    full_name: "",
    password: ""
  });
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [sources, setSources] = useState([]);
  const [status, setStatus] = useState("");

  const isAuthenticated = useMemo(() => Boolean(token), [token]);
  const isAdmin = role === "admin";

  async function handleAuthSubmit(e) {
    e.preventDefault();
    try {
      const data =
        authMode === "login"
          ? await login(authForm.email, authForm.password)
          : await register(
              authForm.email,
              authForm.full_name || "User",
              authForm.password
            );
      setToken(data.access_token);
      setRole(data.role);
      setEmail(data.email);
      localStorage.setItem("token", data.access_token);
      localStorage.setItem("role", data.role);
      localStorage.setItem("email", data.email);
      setStatus("Authenticated successfully.");
    } catch (err) {
      setStatus(err.message);
    }
  }

  function logout() {
    setToken("");
    setRole("");
    setEmail("");
    localStorage.removeItem("token");
    localStorage.removeItem("role");
    localStorage.removeItem("email");
    setMessages([]);
    setSources([]);
  }

  async function sendChat() {
    if (!query.trim()) return;
    const userMessage = { role: "user", content: query };
    setMessages((prev) => [...prev, userMessage]);
    const currentQuery = query;
    setQuery("");
    try {
      const data = await chatQuery(token, currentQuery);
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
      setSources(data.sources || []);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${err.message}` }
      ]);
    }
  }

  async function syncIndex() {
    try {
      setStatus("Syncing index...");
      const data = await adminSync(token);
      setStatus(data.detail);
    } catch (err) {
      setStatus(err.message);
    }
  }

  if (!isAuthenticated) {
    return (
      <div className="container">
        <div className="card auth-card">
          <h1>ENISO Enterprise Assistant</h1>
          <p className="muted">Professional local-first RAG platform</p>
          <div className="tabs">
            <button
              className={authMode === "login" ? "active" : ""}
              onClick={() => setAuthMode("login")}
            >
              Login
            </button>
            <button
              className={authMode === "register" ? "active" : ""}
              onClick={() => setAuthMode("register")}
            >
              Register
            </button>
          </div>
          <form onSubmit={handleAuthSubmit}>
            <input
              placeholder="Email"
              value={authForm.email}
              onChange={(e) => setAuthForm({ ...authForm, email: e.target.value })}
              required
            />
            {authMode === "register" && (
              <input
                placeholder="Full name"
                value={authForm.full_name}
                onChange={(e) =>
                  setAuthForm({ ...authForm, full_name: e.target.value })
                }
                required
              />
            )}
            <input
              placeholder="Password"
              type="password"
              value={authForm.password}
              onChange={(e) => setAuthForm({ ...authForm, password: e.target.value })}
              required
            />
            <button type="submit">Continue</button>
          </form>
          {status && <p className="status">{status}</p>}
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <header className="topbar">
        <div>
          <h2>ENISO Assistant</h2>
          <p className="muted">
            Signed in as {email} ({role})
          </p>
        </div>
        <div className="top-actions">
          {isAdmin && <button onClick={syncIndex}>Sync Index</button>}
          <button onClick={logout}>Logout</button>
        </div>
      </header>

      <main className="chat-shell">
        <section className="chat-pane">
          {messages.map((m, idx) => (
            <div key={idx} className={`msg ${m.role}`}>
              <b>{m.role === "user" ? "You" : "Assistant"}:</b> {m.content}
            </div>
          ))}
        </section>
        <section className="sources-pane">
          <h3>Sources</h3>
          {sources.length === 0 ? (
            <p className="muted">No sources yet.</p>
          ) : (
            sources.map((s) => (
              <span className="pill" key={s}>
                {s}
              </span>
            ))
          )}
        </section>
      </main>

      <footer className="composer">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about ENISO timetable, exams, stage, or PFE..."
          onKeyDown={(e) => e.key === "Enter" && sendChat()}
        />
        <button onClick={sendChat}>Send</button>
      </footer>
      {status && <p className="status">{status}</p>}
    </div>
  );
}
