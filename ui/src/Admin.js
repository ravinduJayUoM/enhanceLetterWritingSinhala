import React, { useState, useEffect, useCallback } from "react";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const ADMIN_USERNAME = "admin@sinhalalipi.lk";
const ADMIN_PASSWORD = "Admin@123";

function getAdminToken() {
  return sessionStorage.getItem("admin_token");
}
function saveAdminToken(t) {
  sessionStorage.setItem("admin_token", t);
}
function clearAdminToken() {
  sessionStorage.removeItem("admin_token");
}

function authHeaders() {
  return { Authorization: `Bearer ${getAdminToken()}` };
}

// ---------------------------------------------------------------------------
// Stat card
// ---------------------------------------------------------------------------
function StatCard({ label, value, sub }) {
  return (
    <div style={{
      background: "#fff", border: "1px solid #e0e0e0", borderRadius: 10,
      padding: "20px 24px", flex: 1, minWidth: 160,
    }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: "#1a237e" }}>{value ?? "—"}</div>
      <div style={{ fontSize: 14, color: "#555", marginTop: 4 }}>{label}</div>
      {sub && <div style={{ fontSize: 12, color: "#888", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Data table
// ---------------------------------------------------------------------------
function DataTable({ columns, rows }) {
  if (!rows || rows.length === 0) {
    return <p style={{ color: "#888", fontSize: 13 }}>දත්ත නොමැත.</p>;
  }
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ background: "#f5f5f5" }}>
            {columns.map((c) => (
              <th key={c.key} style={{ padding: "8px 10px", textAlign: "left",
                borderBottom: "2px solid #e0e0e0", whiteSpace: "nowrap", color: "#37474f" }}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{ borderBottom: "1px solid #f0f0f0",
              background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
              {columns.map((c) => (
                <td key={c.key} style={{ padding: "7px 10px", verticalAlign: "top",
                  maxWidth: c.maxWidth || 200, overflow: "hidden",
                  textOverflow: "ellipsis", whiteSpace: c.wrap ? "normal" : "nowrap" }}>
                  {row[c.key] ?? ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Login screen
// ---------------------------------------------------------------------------
function AdminLogin({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_URL}/admin/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) { setError("Invalid credentials."); return; }
      const data = await res.json();
      saveAdminToken(data.access_token);
      onLogin();
    } catch {
      setError("Connection error.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center",
      justifyContent: "center", background: "#f0f2f9" }}>
      <form onSubmit={handleSubmit} style={{
        background: "#fff", padding: "36px 40px", borderRadius: 12,
        boxShadow: "0 4px 20px rgba(0,0,0,0.1)", width: 360,
      }}>
        <h2 style={{ margin: "0 0 6px", color: "#1a237e" }}>Admin — SinhalaLipi</h2>
        <p style={{ margin: "0 0 24px", fontSize: 13, color: "#888" }}>Admin access only</p>

        {error && <p style={{ color: "#dc2626", fontSize: 13, margin: "0 0 12px" }}>{error}</p>}

        <label style={labelStyle}>Username</label>
        <input value={username} onChange={(e) => setUsername(e.target.value)}
          required style={inputStyle} type="email" autoComplete="username" />

        <label style={labelStyle}>Password</label>
        <input value={password} onChange={(e) => setPassword(e.target.value)}
          required style={inputStyle} type="password" autoComplete="current-password" />

        <button type="submit" disabled={loading} style={btnStyle}>
          {loading ? "Logging in..." : "Login"}
        </button>
      </form>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------
function Dashboard({ onLogout }) {
  const [tab, setTab] = useState("overview");
  const [stats, setStats] = useState(null);
  const [letterRatings, setLetterRatings] = useState([]);
  const [systemFeedback, setSystemFeedback] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchAll = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [sRes, lRes, sfRes] = await Promise.all([
        fetch(`${API_URL}/admin/stats`, { headers: authHeaders() }),
        fetch(`${API_URL}/admin/letter-ratings`, { headers: authHeaders() }),
        fetch(`${API_URL}/admin/system-feedback`, { headers: authHeaders() }),
      ]);
      if (sRes.status === 403 || lRes.status === 403) {
        clearAdminToken(); onLogout(); return;
      }
      setStats(await sRes.json());
      setLetterRatings((await lRes.json()).data || []);
      setSystemFeedback((await sfRes.json()).data || []);
    } catch {
      setError("Failed to load data.");
    } finally {
      setLoading(false);
    }
  }, [onLogout]);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  const downloadCSV = async (endpoint, filename) => {
    const res = await fetch(`${API_URL}${endpoint}`, { headers: authHeaders() });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  };

  const lr = stats?.letter_ratings;
  const sf = stats?.system_feedback;

  return (
    <div style={{ minHeight: "100vh", background: "#f0f2f9", fontFamily: "'Segoe UI', sans-serif" }}>
      {/* Header */}
      <div style={{ background: "#1a237e", color: "#fff", padding: "14px 32px",
        display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ margin: 0, fontSize: 18 }}>SinhalaLipi — Admin Dashboard</h2>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <button onClick={fetchAll} style={ghostBtn}>↻ Refresh</button>
          <button onClick={() => { clearAdminToken(); onLogout(); }} style={ghostBtn}>Logout</button>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: "#fff", borderBottom: "1px solid #e0e0e0",
        padding: "0 32px", display: "flex", gap: 0 }}>
        {[
          { key: "overview",       label: "Overview" },
          { key: "letter_ratings", label: `Letter Ratings (${letterRatings.length})` },
          { key: "system",         label: `System Feedback (${systemFeedback.length})` },
        ].map(({ key, label }) => (
          <button key={key} onClick={() => setTab(key)} style={{
            background: "none", border: "none", padding: "14px 20px",
            fontSize: 14, cursor: "pointer", fontWeight: tab === key ? 700 : 400,
            color: tab === key ? "#1a237e" : "#555",
            borderBottom: tab === key ? "3px solid #1a237e" : "3px solid transparent",
          }}>
            {label}
          </button>
        ))}
      </div>

      <div style={{ padding: "28px 32px" }}>
        {loading && <p style={{ color: "#888" }}>Loading...</p>}
        {error && <p style={{ color: "#dc2626" }}>{error}</p>}

        {/* Overview tab */}
        {tab === "overview" && stats && (
          <>
            <h3 style={sectionTitle}>Users</h3>
            <div style={cardRow}>
              <StatCard label="Registered Users" value={stats.users?.total} />
            </div>

            <h3 style={sectionTitle}>Letter Ratings</h3>
            <div style={cardRow}>
              <StatCard label="Total Ratings" value={lr?.total} />
              <StatCard label="Avg Overall Quality" value={lr?.avg_overall} sub="/ 5" />
              <StatCard label="Avg Matches Request" value={lr?.avg_match} sub="/ 5" />
              <StatCard label="Avg Language Quality" value={lr?.avg_language} sub="/ 5" />
              <StatCard label="Avg Structure" value={lr?.avg_structure} sub="/ 5" />
            </div>

            <h3 style={sectionTitle}>System Usability Feedback</h3>
            <div style={cardRow}>
              <StatCard label="Total Responses" value={sf?.total} />
              <StatCard label="Avg Ease of Use" value={sf?.avg_ease_of_use} sub="/ 5" />
              <StatCard label="Avg Ease of Describing" value={sf?.avg_ease_of_describing} sub="/ 5" />
              <StatCard label="Avg Gap Questions" value={sf?.avg_gap_questions} sub="/ 5" />
              <StatCard label="Avg Confidence" value={sf?.avg_confidence} sub="/ 5" />
              <StatCard label="Avg Would Use Again" value={sf?.avg_would_use_again} sub="/ 5" />
            </div>
          </>
        )}

        {/* Letter ratings tab */}
        {tab === "letter_ratings" && (
          <>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h3 style={{ margin: 0, color: "#1a237e" }}>Letter Ratings</h3>
              <button onClick={() => downloadCSV("/admin/export/letter-ratings", "letter_ratings.csv")} style={exportBtn}>
                ↓ Export CSV
              </button>
            </div>
            <DataTable
              rows={letterRatings}
              columns={[
                { key: "id",                label: "ID",        maxWidth: 50 },
                { key: "timestamp",         label: "Time",      maxWidth: 140 },
                { key: "username",          label: "User",      maxWidth: 120 },
                { key: "letter_category",   label: "Category",  maxWidth: 100 },
                { key: "quality_overall",   label: "Overall",   maxWidth: 70 },
                { key: "quality_match",     label: "Match",     maxWidth: 60 },
                { key: "quality_language",  label: "Language",  maxWidth: 70 },
                { key: "quality_structure", label: "Structure", maxWidth: 70 },
                { key: "comments",          label: "Comments",  maxWidth: 260, wrap: true },
                { key: "original_prompt",   label: "Prompt",    maxWidth: 260, wrap: true },
              ]}
            />
          </>
        )}

        {/* System feedback tab */}
        {tab === "system" && (
          <>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h3 style={{ margin: 0, color: "#1a237e" }}>System Feedback</h3>
              <button onClick={() => downloadCSV("/admin/export/system-feedback", "system_feedback.csv")} style={exportBtn}>
                ↓ Export CSV
              </button>
            </div>
            <DataTable
              rows={systemFeedback}
              columns={[
                { key: "id",                    label: "ID",           maxWidth: 50 },
                { key: "timestamp",             label: "Time",         maxWidth: 140 },
                { key: "username",              label: "User",         maxWidth: 120 },
                { key: "ease_of_use",           label: "Ease of Use",  maxWidth: 80 },
                { key: "ease_of_describing",    label: "Describing",   maxWidth: 80 },
                { key: "gap_questions_helpful", label: "Gap Qs",       maxWidth: 60 },
                { key: "confidence_in_output",  label: "Confidence",   maxWidth: 80 },
                { key: "would_use_again",       label: "Use Again",    maxWidth: 70 },
                { key: "liked_most",            label: "Liked",        maxWidth: 200, wrap: true },
                { key: "needs_improvement",     label: "Improve",      maxWidth: 200, wrap: true },
                { key: "issues_faced",          label: "Issues",       maxWidth: 200, wrap: true },
              ]}
            />
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------
const labelStyle = { display: "block", fontSize: 13, color: "#555", marginBottom: 5, marginTop: 14 };
const inputStyle = {
  width: "100%", padding: "9px 12px", borderRadius: 6,
  border: "1px solid #ccc", fontSize: 14, boxSizing: "border-box",
};
const btnStyle = {
  marginTop: 22, width: "100%", padding: "11px", background: "#1a237e",
  color: "#fff", border: "none", borderRadius: 6, fontSize: 15,
  fontWeight: 600, cursor: "pointer",
};
const ghostBtn = {
  background: "transparent", border: "1px solid rgba(255,255,255,0.5)",
  color: "#fff", borderRadius: 6, padding: "6px 14px",
  cursor: "pointer", fontSize: 13,
};
const exportBtn = {
  background: "#1a237e", color: "#fff", border: "none", borderRadius: 6,
  padding: "7px 16px", fontSize: 13, cursor: "pointer", fontWeight: 600,
};
const sectionTitle = { color: "#37474f", marginTop: 28, marginBottom: 12 };
const cardRow = { display: "flex", gap: 16, flexWrap: "wrap" };

// ---------------------------------------------------------------------------
// Root export
// ---------------------------------------------------------------------------
export default function Admin() {
  const [loggedIn, setLoggedIn] = useState(!!getAdminToken());

  if (!loggedIn) {
    return <AdminLogin onLogin={() => setLoggedIn(true)} />;
  }
  return <Dashboard onLogout={() => setLoggedIn(false)} />;
}
