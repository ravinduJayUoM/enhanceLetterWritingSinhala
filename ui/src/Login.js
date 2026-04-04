import React, { useState } from "react";
import { saveToken, saveProfile } from "./auth";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const styles = {
  container: {
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#f0f4ff",
    fontFamily: "'Segoe UI', 'Noto Sans Sinhala', sans-serif",
  },
  card: {
    background: "#fff",
    borderRadius: 12,
    padding: "40px 36px",
    width: 360,
    boxShadow: "0 4px 24px rgba(0,0,0,0.10)",
  },
  title: { textAlign: "center", color: "#1a237e", marginBottom: 8, fontSize: 22 },
  subtitle: { textAlign: "center", color: "#666", marginBottom: 28, fontSize: 14 },
  label: { display: "block", marginBottom: 4, color: "#333", fontSize: 14, fontWeight: 500 },
  input: {
    width: "100%", padding: "9px 12px", borderRadius: 6, fontSize: 14,
    border: "1px solid #ccc", boxSizing: "border-box", marginBottom: 16,
    outline: "none",
  },
  button: {
    width: "100%", padding: "10px 0", background: "#1a237e", color: "#fff",
    border: "none", borderRadius: 6, fontSize: 15, fontWeight: 600,
    cursor: "pointer", marginTop: 4,
  },
  error: { color: "#c62828", fontSize: 13, marginBottom: 12, textAlign: "center" },
  link: { color: "#1a237e", cursor: "pointer", textDecoration: "underline", fontSize: 14 },
  footer: { textAlign: "center", marginTop: 20, color: "#555", fontSize: 14 },
};

export default function Login({ onLogin, onGoRegister }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) {
        const data = await res.json();
        setError(data.detail || "Login failed.");
        return;
      }
      const { access_token } = await res.json();
      saveToken(access_token);

      // Fetch profile
      const profileRes = await fetch(`${API_URL}/auth/me`, {
        headers: { Authorization: `Bearer ${access_token}` },
      });
      const profile = await profileRes.json();
      saveProfile(profile);
      onLogin(profile);
    } catch {
      setError("Could not connect to server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h2 style={styles.title}>SinhalaLipi</h2>
        <p style={styles.subtitle}>ඔබගේ ගිණුමට ප්‍රවේශ වන්න</p>

        {error && <div style={styles.error}>{error}</div>}

        <form onSubmit={handleSubmit}>
          <label style={styles.label}>Username</label>
          <input
            style={styles.input}
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            autoFocus
          />
          <label style={styles.label}>Password</label>
          <input
            style={styles.input}
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button style={styles.button} type="submit" disabled={loading}>
            {loading ? "Logging in…" : "Login"}
          </button>
        </form>

        <div style={styles.footer}>
          ගිණුමක් නැද්ද?{" "}
          <span style={styles.link} onClick={onGoRegister}>
            Register
          </span>
        </div>
      </div>
    </div>
  );
}
