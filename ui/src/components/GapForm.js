import React, { useState } from "react";

export default function GapForm({ fields, onSubmit }) {
  const [answers, setAnswers] = useState(
    Object.fromEntries(fields.map((f) => [f.field, ""]))
  );

  const set = (field) => (e) =>
    setAnswers((prev) => ({ ...prev, [field]: e.target.value }));

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(answers);
  };

  const allFilled = fields.every((f) => answers[f.field].trim() !== "");

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        background: "#f8f9ff",
        border: "1px solid #dde",
        borderRadius: 10,
        padding: "20px 20px 16px",
        marginTop: 12,
      }}
    >
      <p style={{ margin: "0 0 14px", fontSize: 14, color: "#444", fontWeight: 500 }}>
        පහත විස්තර සපයන්න:
      </p>

      {fields.map(({ field, question }) => (
        <div key={field} style={{ marginBottom: 14 }}>
          <label
            style={{
              display: "block",
              fontSize: 13,
              color: "#333",
              marginBottom: 5,
              fontFamily: "'Segoe UI', 'Noto Sans Sinhala', sans-serif",
            }}
          >
            {question}
          </label>
          <input
            type="text"
            value={answers[field]}
            onChange={set(field)}
            required
            style={{
              width: "100%",
              padding: "8px 10px",
              borderRadius: 6,
              border: "1px solid #ccc",
              fontSize: 14,
              boxSizing: "border-box",
              fontFamily: "'Segoe UI', 'Noto Sans Sinhala', sans-serif",
            }}
          />
        </div>
      ))}

      <button
        type="submit"
        disabled={!allFilled}
        style={{
          marginTop: 4,
          padding: "9px 24px",
          background: allFilled ? "#1a237e" : "#aaa",
          color: "#fff",
          border: "none",
          borderRadius: 6,
          fontSize: 14,
          fontWeight: 600,
          cursor: allFilled ? "pointer" : "not-allowed",
          width: "100%",
        }}
      >
        ඉදිරියට යන්න →
      </button>
    </form>
  );
}
