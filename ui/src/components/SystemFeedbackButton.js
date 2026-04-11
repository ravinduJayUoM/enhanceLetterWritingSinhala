import React, { useState } from "react";
import { getToken } from "../auth";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const LIKERT_QUESTIONS = [
  { key: "ease_of_use",           label: "පද්ධතිය භාවිත කිරීම පහසුයි" },
  { key: "ease_of_describing",    label: "ලිපිය විස්තර කිරීම පහසුයි" },
  { key: "gap_questions_helpful", label: "අතිරේක ප්‍රශ්න (Gap Form) ප්‍රයෝජනවත්" },
  { key: "confidence_in_output",  label: "ජනනය වූ ලිපිය නිවැරදි යැයි විශ්වාසයි" },
  { key: "would_use_again",       label: "මෙම පද්ධතිය නැවත භාවිත කරමි" },
];

function LikertRow({ label, value, onChange }) {
  const [hovered, setHovered] = useState(0);
  const LABELS = ["", "දෙකමත් නොකැමතියි", "කැමති නෑ", "මධ්‍යම", "කැමතියි", "ඉතා කැමතියි"];
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 13, color: "#333", marginBottom: 5,
        fontFamily: "'Segoe UI', 'Noto Sans Sinhala', sans-serif" }}>
        {label}
      </div>
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            type="button"
            onMouseEnter={() => setHovered(n)}
            onMouseLeave={() => setHovered(0)}
            onClick={() => onChange(n)}
            style={{
              width: 36, height: 36, borderRadius: 6, border: "1px solid",
              fontSize: 15, cursor: "pointer", transition: "all 0.12s",
              borderColor: n <= (hovered || value) ? "#1a237e" : "#ccc",
              background: n <= (hovered || value) ? "#1a237e" : "#fff",
              color: n <= (hovered || value) ? "#fff" : "#999",
              fontWeight: 600,
            }}
          >
            {n}
          </button>
        ))}
        <span style={{ marginLeft: 6, fontSize: 11, color: "#888" }}>
          {value ? LABELS[value] : ""}
        </span>
      </div>
    </div>
  );
}

export default function SystemFeedbackButton() {
  const [open, setOpen] = useState(false);
  const [scores, setScores] = useState({
    ease_of_use: 0,
    ease_of_describing: 0,
    gap_questions_helpful: 0,
    confidence_in_output: 0,
    would_use_again: 0,
  });
  const [text, setText] = useState({ liked_most: "", needs_improvement: "", issues_faced: "" });
  const [status, setStatus] = useState(null); // null | "saving" | "done" | "error"

  const set = (key) => (val) => setScores((prev) => ({ ...prev, [key]: val }));
  const setText_ = (key) => (e) => setText((prev) => ({ ...prev, [key]: e.target.value }));

  const allScored = LIKERT_QUESTIONS.every((q) => scores[q.key] > 0);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setStatus("saving");
    try {
      const res = await fetch(`${API_URL}/feedback/system/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${getToken()}`,
        },
        body: JSON.stringify({ ...scores, ...text }),
      });
      if (!res.ok) throw new Error();
      setStatus("done");
      setTimeout(() => { setOpen(false); setStatus(null); }, 2000);
    } catch {
      setStatus("error");
    }
  };

  return (
    <>
      {/* Fixed bottom-right button */}
      <button
        onClick={() => setOpen(true)}
        title="Rate this system"
        style={{
          position: "fixed", bottom: 28, right: 28, zIndex: 1000,
          background: "#1a237e", color: "#fff",
          border: "none", borderRadius: 50, width: 52, height: 52,
          fontSize: 22, cursor: "pointer", boxShadow: "0 4px 12px rgba(0,0,0,0.25)",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}
      >
        💬
      </button>

      {/* Modal overlay */}
      {open && (
        <div
          onClick={() => setOpen(false)}
          style={{
            position: "fixed", inset: 0, background: "rgba(0,0,0,0.45)",
            zIndex: 1001, display: "flex", alignItems: "center", justifyContent: "center",
            padding: 16,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: "#fff", borderRadius: 12, padding: "28px 28px 22px",
              maxWidth: 520, width: "100%", maxHeight: "90vh", overflowY: "auto",
              boxShadow: "0 8px 32px rgba(0,0,0,0.18)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 18 }}>
              <div>
                <h3 style={{ margin: 0, color: "#1a237e", fontSize: 17 }}>පද්ධතිය ගැන ඔබේ අදහස</h3>
                <p style={{ margin: "4px 0 0", fontSize: 12, color: "#888" }}>
                  ඔබේ ප්‍රතිපෝෂණය පද්ධතිය වැඩිදියුණු කිරීමට ඉතා වැදගත්.
                </p>
              </div>
              <button
                onClick={() => setOpen(false)}
                style={{ background: "none", border: "none", fontSize: 20, cursor: "pointer", color: "#888", lineHeight: 1 }}
              >
                ×
              </button>
            </div>

            {status === "done" ? (
              <p style={{ color: "#16a34a", fontWeight: 600, textAlign: "center", padding: "20px 0" }}>
                ✅ ස්තූතියි! ඔබේ ප්‍රතිපෝෂණය සාර්ථකව සුරැකිණි.
              </p>
            ) : status === "error" ? (
              <p style={{ color: "#dc2626", textAlign: "center" }}>❌ දෝෂයකි. කරුණාකර නැවත උත්සාහ කරන්න.</p>
            ) : (
              <form onSubmit={handleSubmit}>
                {LIKERT_QUESTIONS.map((q) => (
                  <LikertRow key={q.key} label={q.label} value={scores[q.key]} onChange={set(q.key)} />
                ))}

                <hr style={{ border: "none", borderTop: "1px solid #eee", margin: "18px 0 14px" }} />

                {[
                  { key: "liked_most",         label: "වඩාත් කැමති දේ කුමක්ද?" },
                  { key: "needs_improvement",  label: "වැඩිදියුණු විය යුතු දේ?" },
                  { key: "issues_faced",        label: "ඔබ අත්දුටු ගැටලු?" },
                ].map(({ key, label }) => (
                  <div key={key} style={{ marginBottom: 12 }}>
                    <label style={{ display: "block", fontSize: 13, color: "#444", marginBottom: 4,
                      fontFamily: "'Segoe UI', 'Noto Sans Sinhala', sans-serif" }}>
                      {label}
                    </label>
                    <textarea
                      value={text[key]}
                      onChange={setText_(key)}
                      rows={2}
                      style={{
                        width: "100%", padding: "7px 10px", borderRadius: 6,
                        border: "1px solid #ccc", fontSize: 13, resize: "vertical",
                        boxSizing: "border-box",
                        fontFamily: "'Segoe UI', 'Noto Sans Sinhala', sans-serif",
                      }}
                    />
                  </div>
                ))}

                <button
                  type="submit"
                  disabled={!allScored || status === "saving"}
                  style={{
                    marginTop: 6, padding: "10px 24px", width: "100%",
                    background: allScored ? "#1a237e" : "#aaa",
                    color: "#fff", border: "none", borderRadius: 6,
                    fontSize: 14, fontWeight: 600,
                    cursor: allScored ? "pointer" : "not-allowed",
                  }}
                >
                  {status === "saving" ? "සුරකිමින්..." : "ඉදිරිපත් කරන්න"}
                </button>
              </form>
            )}
          </div>
        </div>
      )}
    </>
  );
}
