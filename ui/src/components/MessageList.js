import React, { useEffect, useRef } from "react";

const styles = {
  container: {
    border: "1px solid #ddd",
    borderRadius: 10,
    padding: 16,
    minHeight: 320,
    maxHeight: 480,
    overflowY: "auto",
    background: "#fafafa",
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  bubble: {
    display: "inline-block",
    padding: "10px 14px",
    borderRadius: 16,
    maxWidth: "78%",
    wordBreak: "break-word",
    lineHeight: 1.5,
    fontSize: 15,
  },
  userRow: { textAlign: "right" },
  systemRow: { textAlign: "left" },
  userBubble: { background: "#d1ecf1", borderBottomRightRadius: 4 },
  systemBubble: { background: "#e8f5e9", borderBottomLeftRadius: 4 },
  waitingBubble: { background: "#fff9c4", borderBottomLeftRadius: 4, color: "#888" },
};

export default function MessageList({ messages, waiting }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, waiting]);

  return (
    <div style={styles.container}>
      {messages.map((msg, i) => (
        <div key={i} style={msg.sender === "user" ? styles.userRow : styles.systemRow}>
          <span
            style={{
              ...styles.bubble,
              ...(msg.sender === "user" ? styles.userBubble : styles.systemBubble),
            }}
          >
            {msg.text}
          </span>
        </div>
      ))}
      {waiting && (
        <div style={styles.systemRow}>
          <span style={{ ...styles.bubble, ...styles.waitingBubble }}>
            ⏳ කරුණාකර රැඳී සිටින්න…
          </span>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
