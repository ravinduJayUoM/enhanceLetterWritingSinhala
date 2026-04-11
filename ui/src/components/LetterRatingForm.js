import React, { useState } from "react";

const QUESTIONS = [
  { key: "quality_overall",   label: "ලිපියේ සමස්ත ගුණාත්මකභාවය" },
  { key: "quality_match",     label: "ලිපිය ඔබේ ඉල්ලීමට ගැලපේ" },
  { key: "quality_language",  label: "සිංහල භාෂාව ස්වාභාවිකයි" },
  { key: "quality_structure", label: "විධිමත් ලිපි ආකෘතිය නිවැරදියි" },
];

function LikertRow({ label, value, onChange }) {
  const [hovered, setHovered] = useState(0);
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 13, color: "#444", marginBottom: 5,
        fontFamily: "'Segoe UI', 'Noto Sans Sinhala', sans-serif" }}>
        {label}
      </div>
      <div style={{ display: "flex", gap: 6 }}>
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            type="button"
            onMouseEnter={() => setHovered(n)}
            onMouseLeave={() => setHovered(0)}
            onClick={() => onChange(n)}
            style={{
              width: 36, height: 36, borderRadius: 6, border: "1px solid",
              fontSize: 18, cursor: "pointer", transition: "all 0.12s",
              borderColor: n <= (hovered || value) ? "#1a237e" : "#ccc",
              background: n <= (hovered || value) ? "#1a237e" : "#fff",
              color: n <= (hovered || value) ? "#fff" : "#999",
              fontWeight: 600,
            }}
          >
            {n}
          </button>
        ))}
        <span style={{ marginLeft: 6, fontSize: 12, color: "#888", alignSelf: "center" }}>
          {value ? ["", "දුර්වලයි", "සාමාන්‍යයි", "හොඳයි", "ඉතා හොඳයි", "විශිෂ්ටයි"][value] : ""}
        </span>
      </div>
    </div>
  );
}

export default function LetterRatingForm({ onSubmit, status }) {
  const [scores, setScores] = useState({
    quality_overall: 0,
    quality_match: 0,
    quality_language: 0,
    quality_structure: 0,
  });
  const [comments, setComments] = useState("");

  const set = (key) => (val) => setScores((prev) => ({ ...prev, [key]: val }));

  const allScored = QUESTIONS.every((q) => scores[q.key] > 0);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({ ...scores, comments });
  };

  if (status === "saving") {
    return <p style={{ color: "#6b7280", margin: 0 }}>⏳ ශ්‍රේණිගත කිරීම සුරකිමින්...</p>;
  }

  if (status === "done") {
    return (
      <p style={{ color: "#16a34a", fontWeight: 600, margin: 0 }}>
        ✅ ස්තූතියි! ඔබේ ශ්‍රේණිගත කිරීම සාර්ථකව සුරැකිණි.
      </p>
    );
  }

  if (status === "indexed") {
    return (
      <p style={{ color: "#16a34a", fontWeight: 600, margin: 0 }}>
        ✅ ස්තූතියි! ලිපිය දත්ත ගබඩාවට ද සාර්ථකව එකතු කරන ලදී.
      </p>
    );
  }

  if (status === "error") {
    return (
      <p style={{ color: "#dc2626", margin: 0 }}>❌ දෝෂයකි. කරුණාකර නැවත උත්සාහ කරන්න.</p>
    );
  }

  return (
    <form onSubmit={handleSubmit}>
      <p style={{ margin: "0 0 14px", fontWeight: 600, color: "#37474f", fontSize: 14 }}>
        මෙම ලිපිය ගැන ඔබේ අදහස ලබා දෙන්න:
      </p>
      {QUESTIONS.map((q) => (
        <LikertRow key={q.key} label={q.label} value={scores[q.key]} onChange={set(q.key)} />
      ))}
      <div style={{ marginBottom: 14 }}>
        <label style={{ display: "block", fontSize: 13, color: "#444", marginBottom: 5,
          fontFamily: "'Segoe UI', 'Noto Sans Sinhala', sans-serif" }}>
          අමතර අදහස් (අනිවාර්ය නොවේ)
        </label>
        <textarea
          value={comments}
          onChange={(e) => setComments(e.target.value)}
          rows={2}
          style={{
            width: "100%", padding: "8px 10px", borderRadius: 6,
            border: "1px solid #ccc", fontSize: 13, resize: "vertical",
            boxSizing: "border-box",
            fontFamily: "'Segoe UI', 'Noto Sans Sinhala', sans-serif",
          }}
        />
      </div>
      <button
        type="submit"
        disabled={!allScored}
        style={{
          padding: "9px 24px",
          background: allScored ? "#1a237e" : "#aaa",
          color: "#fff", border: "none", borderRadius: 6,
          fontSize: 14, fontWeight: 600,
          cursor: allScored ? "pointer" : "not-allowed",
          width: "100%",
        }}
      >
        ශ්‍රේණිගත කිරීම ඉදිරිපත් කරන්න
      </button>
    </form>
  );
}
