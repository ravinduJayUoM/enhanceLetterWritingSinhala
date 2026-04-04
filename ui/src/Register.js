import React, { useState } from "react";

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
    padding: "36px 36px",
    width: 400,
    boxShadow: "0 4px 24px rgba(0,0,0,0.10)",
  },
  title: { textAlign: "center", color: "#1a237e", marginBottom: 6, fontSize: 22 },
  subtitle: { textAlign: "center", color: "#666", marginBottom: 24, fontSize: 14 },
  label: { display: "block", marginBottom: 4, color: "#333", fontSize: 13, fontWeight: 500 },
  input: {
    width: "100%", padding: "8px 12px", borderRadius: 6, fontSize: 14,
    border: "1px solid #ccc", boxSizing: "border-box", marginBottom: 14,
    outline: "none",
  },
  sectionTitle: { color: "#1a237e", fontSize: 13, fontWeight: 600, marginBottom: 10, marginTop: 4 },
  button: {
    width: "100%", padding: "10px 0", background: "#1a237e", color: "#fff",
    border: "none", borderRadius: 6, fontSize: 15, fontWeight: 600,
    cursor: "pointer", marginTop: 4,
  },
  error: { color: "#c62828", fontSize: 13, marginBottom: 12, textAlign: "center" },
  success: { color: "#2e7d32", fontSize: 13, marginBottom: 12, textAlign: "center" },
  link: { color: "#1a237e", cursor: "pointer", textDecoration: "underline", fontSize: 14 },
  footer: { textAlign: "center", marginTop: 20, color: "#555", fontSize: 14 },
  optional: { color: "#999", fontWeight: 400, fontSize: 12 },
};

export default function Register({ onGoLogin }) {
  const [form, setForm] = useState({
    username: "", password: "", confirmPassword: "",
    full_name: "", title: "", address_line1: "", address_line2: "", phone: "",
  });
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);

  const set = (field) => (e) => setForm((f) => ({ ...f, [field]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (form.password !== form.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    if (form.password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: form.username,
          password: form.password,
          full_name: form.full_name,
          title: form.title,
          address_line1: form.address_line1,
          address_line2: form.address_line2,
          phone: form.phone,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail || "Registration failed.");
        return;
      }
      setSuccess("Account created! You can now log in.");
      setTimeout(onGoLogin, 1500);
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
        <p style={styles.subtitle}>නව ගිණුමක් සාදන්න</p>

        {error && <div style={styles.error}>{error}</div>}
        {success && <div style={styles.success}>{success}</div>}

        <form onSubmit={handleSubmit}>
          <div style={styles.sectionTitle}>Account Details</div>

          <label style={styles.label}>Username</label>
          <input style={styles.input} type="text" value={form.username} onChange={set("username")} required autoFocus />

          <label style={styles.label}>Password</label>
          <input style={styles.input} type="password" value={form.password} onChange={set("password")} required />

          <label style={styles.label}>Confirm Password</label>
          <input style={styles.input} type="password" value={form.confirmPassword} onChange={set("confirmPassword")} required />

          <div style={{ ...styles.sectionTitle, marginTop: 16 }}>Personal Details <span style={styles.optional}>(used in letter header)</span></div>

          <label style={styles.label}>Full Name</label>
          <input style={styles.input} type="text" value={form.full_name} onChange={set("full_name")} required />

          <label style={styles.label}>Title / Position <span style={styles.optional}>(optional)</span></label>
          <input style={styles.input} type="text" placeholder="e.g. Principal, Manager" value={form.title} onChange={set("title")} />

          <label style={styles.label}>Address Line 1 <span style={styles.optional}>(optional)</span></label>
          <input style={styles.input} type="text" value={form.address_line1} onChange={set("address_line1")} />

          <label style={styles.label}>Address Line 2 <span style={styles.optional}>(optional)</span></label>
          <input style={styles.input} type="text" value={form.address_line2} onChange={set("address_line2")} />

          <label style={styles.label}>Phone <span style={styles.optional}>(optional)</span></label>
          <input style={styles.input} type="text" value={form.phone} onChange={set("phone")} />

          <button style={styles.button} type="submit" disabled={loading}>
            {loading ? "Creating account…" : "Register"}
          </button>
        </form>

        <div style={styles.footer}>
          දැනටමත් ගිණුමක් ඇද්ද?{" "}
          <span style={styles.link} onClick={onGoLogin}>Login</span>
        </div>
      </div>
    </div>
  );
}
