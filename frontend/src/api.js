const API_BASE = "http://localhost:8000/api";

export async function login(email, password) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password })
  });
  if (!res.ok) throw new Error("Login failed");
  return res.json();
}

export async function register(email, full_name, password, role = "student") {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, full_name, password, role })
  });
  if (!res.ok) throw new Error("Register failed");
  return res.json();
}

export async function chatQuery(token, query) {
  const res = await fetch(`${API_BASE}/chat/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`
    },
    body: JSON.stringify({ query })
  });
  if (!res.ok) throw new Error("Chat request failed");
  return res.json();
}

export async function adminSync(token) {
  const res = await fetch(`${API_BASE}/admin/sync-index`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`
    }
  });
  if (!res.ok) throw new Error("Sync failed");
  return res.json();
}
