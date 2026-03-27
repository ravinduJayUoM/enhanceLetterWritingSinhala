import React, { useState, useEffect } from "react";

function StarRating({ onRate }) {
  const [hovered, setHovered] = useState(0);
  const [selected, setSelected] = useState(0);

  const handleClick = (star) => {
    setSelected(star);
    onRate(star);
  };

  return (
    <div style={{ display: "flex", gap: 6, cursor: "pointer", fontSize: 32 }}>
      {[1, 2, 3, 4, 5].map((star) => (
        <span
          key={star}
          onMouseEnter={() => setHovered(star)}
          onMouseLeave={() => setHovered(0)}
          onClick={() => handleClick(star)}
          style={{
            color: star <= (hovered || selected) ? "#f59e0b" : "#d1d5db",
            transition: "color 0.15s",
            pointerEvents: selected ? "none" : "auto",
          }}
        >
          ★
        </span>
      ))}
    </div>
  );
}

const styles = {
  wrapper: {
    marginTop: 24,
    border: "1px solid #b0bec5",
    borderRadius: 10,
    overflow: "hidden",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    background: "#37474f",
    color: "#fff",
    padding: "10px 16px",
    fontSize: 15,
    fontWeight: 600,
  },
  copyBtn: {
    background: "transparent",
    border: "1px solid #90a4ae",
    borderRadius: 6,
    color: "#cfd8dc",
    padding: "4px 12px",
    cursor: "pointer",
    fontSize: 13,
  },
  copyBtnDone: {
    color: "#a5d6a7",
    borderColor: "#a5d6a7",
  },
  editBtn: {
    background: "transparent",
    border: "1px solid #90a4ae",
    borderRadius: 6,
    color: "#cfd8dc",
    padding: "4px 12px",
    cursor: "pointer",
    fontSize: 13,
    marginRight: 8,
  },
  saveBtn: {
    background: "#43a047",
    border: "none",
    borderRadius: 6,
    color: "#fff",
    padding: "4px 14px",
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 600,
    marginRight: 8,
  },
  cancelBtn: {
    background: "transparent",
    border: "1px solid #ef9a9a",
    borderRadius: 6,
    color: "#ef9a9a",
    padding: "4px 12px",
    cursor: "pointer",
    fontSize: 13,
    marginRight: 8,
  },
  body: {
    background: "#fff",
    padding: 20,
    whiteSpace: "pre-wrap",
    lineHeight: 1.8,
    fontSize: 15,
    fontFamily: "'Noto Sans Sinhala', 'Iskoola Pota', sans-serif",
    color: "#212121",
    maxHeight: 520,
    overflowY: "auto",
  },
  textarea: {
    width: "100%",
    minHeight: 380,
    padding: 20,
    lineHeight: 1.8,
    fontSize: 15,
    fontFamily: "'Noto Sans Sinhala', 'Iskoola Pota', sans-serif",
    color: "#212121",
    background: "#fffde7",
    border: "none",
    outline: "none",
    resize: "vertical",
    boxSizing: "border-box",
    display: "block",
  },
};

export default function LetterDisplay({ letter, onRate, ratingStatus }) {
  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editedLetter, setEditedLetter] = useState(letter);

  // Reset edited content whenever a new letter arrives
  useEffect(() => {
    setEditedLetter(letter);
    setIsEditing(false);
  }, [letter]);

  const handleCopy = () => {
    navigator.clipboard.writeText(editedLetter).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleSave = () => {
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditedLetter(letter);
    setIsEditing(false);
  };

  if (!letter) return null;

  return (
    <div style={styles.wrapper}>
      <div style={styles.header}>
        <span>📄 ජනනය වූ ලිපිය</span>
        <div style={{ display: "flex", alignItems: "center" }}>
          {isEditing ? (
            <>
              <button style={styles.saveBtn} onClick={handleSave}>
                ✓ සුරකින්න
              </button>
              <button style={styles.cancelBtn} onClick={handleCancel}>
                අවලංගු කරන්න
              </button>
            </>
          ) : (
            <button style={styles.editBtn} onClick={() => setIsEditing(true)}>
              ✏️ සංස්කරණය
            </button>
          )}
          <button
            style={{ ...styles.copyBtn, ...(copied ? styles.copyBtnDone : {}) }}
            onClick={handleCopy}
          >
            {copied ? "✓ පිටපත් විය" : "පිටපත් කරන්න"}
          </button>
        </div>
      </div>
      {isEditing ? (
        <textarea
          style={styles.textarea}
          value={editedLetter}
          onChange={(e) => setEditedLetter(e.target.value)}
          autoFocus
        />
      ) : (
        <div style={styles.body}>{editedLetter}</div>
      )}
      <div style={{ padding: "16px 20px", borderTop: "1px solid #eceff1", background: "#fafafa" }}>
        <p style={{ margin: "0 0 10px", fontWeight: 600, color: "#37474f" }}>මෙම ලිපිය ගැන ඔබේ අදහස (rating) ලබා දෙන්න.</p>
        {ratingStatus === null && <StarRating onRate={onRate} />}
        {ratingStatus === "saving" && (
          <span style={{ color: "#6b7280" }}>⏳ ශ්‍රේණිගත කිරීම සුරකිමින්...</span>
        )}
        {ratingStatus === "indexed" && (
          <span style={{ color: "#16a34a", fontWeight: 600 }}>✅ ස්තූතියි! ලිපිය දත්ත ගබඩාවට සාර්ථකව එකතු කරන ලදී.</span>
        )}
        {ratingStatus === "saved" && (
          <span style={{ color: "#2563eb", fontWeight: 600 }}>✅ ස්තූතිවන්ත යි! ඔබගේ ශ්‍රේණිගත කිරීම සුරැකිණි.</span>
        )}
        {ratingStatus === "error" && (
          <span style={{ color: "#dc2626" }}>❌ දෝෂයකි. කරුණාකර නැවත උත්සාහ කරන්න.</span>
        )}
      </div>
    </div>
  );
}
