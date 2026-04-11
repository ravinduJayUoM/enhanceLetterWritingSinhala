import React, { useState, useEffect } from "react";
import LetterRatingForm from "./LetterRatingForm";

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
  // onRate now receives the full rating object {quality_overall, ...}
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
        <LetterRatingForm onSubmit={onRate} status={ratingStatus} />
      </div>
    </div>
  );
}
