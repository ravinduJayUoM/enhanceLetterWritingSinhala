import React, { useRef, useEffect } from "react";

const styles = {
  wrapper: {
    display: "flex",
    gap: 8,
    marginTop: 12,
  },
  input: {
    flex: 1,
    padding: "10px 14px",
    borderRadius: 24,
    border: "1px solid #ccc",
    fontSize: 15,
    outline: "none",
    fontFamily: "inherit",
  },
  button: {
    padding: "10px 20px",
    borderRadius: 24,
    border: "none",
    background: "#1976d2",
    color: "#fff",
    fontSize: 15,
    cursor: "pointer",
    fontFamily: "inherit",
    transition: "background 0.2s",
  },
  buttonDisabled: {
    background: "#90caf9",
    cursor: "not-allowed",
  },
};

export default function InputBar({ value, onChange, onSend, disabled, placeholder }) {
  const inputRef = useRef(null);

  // Keep focus on the input whenever it becomes enabled
  useEffect(() => {
    if (!disabled) inputRef.current?.focus();
  }, [disabled]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !disabled) onSend();
  };

  return (
    <div style={styles.wrapper}>
      <input
        ref={inputRef}
        type="text"
        style={styles.input}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder={placeholder || "සිංහලෙන් ඇතුළත් කරන්න…"}
      />
      <button
        style={{ ...styles.button, ...(disabled ? styles.buttonDisabled : {}) }}
        onClick={onSend}
        disabled={disabled}
      >
        යවන්න
      </button>
    </div>
  );
}
