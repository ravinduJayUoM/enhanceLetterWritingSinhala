import React, { useState } from "react";

const API_URL = "http://localhost:8000";

function Chat() {
  const [messages, setMessages] = useState([
    { sender: "system", text: "ඔබගේ ලිපි ඉල්ලීම සිංහලෙන් ඇතුළත් කරන්න." }
  ]);
  const [input, setInput] = useState("");
  const [waiting, setWaiting] = useState(false);
  const [letter, setLetter] = useState(null);

  // Store enhanced prompt for letter generation
  const [enhancedPrompt, setEnhancedPrompt] = useState("");
  const [originalPrompt, setOriginalPrompt] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;
    setMessages((msgs) => [...msgs, { sender: "user", text: input }]);
    setWaiting(true);
    setOriginalPrompt(input);

    // Send to /process_query/
    const res = await fetch(`${API_URL}/process_query/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: input })
    });
    const data = await res.json();

    if (data.status === "incomplete") {
      // Ask missing info questions
      for (const [field, question] of Object.entries(data.questions)) {
        setMessages((msgs) => [
          ...msgs,
          { sender: "system", text: question }
        ]);
        // Wait for user input for each question
        let answer = await new Promise((resolve) => {
          const handler = (e) => {
            if (e.key === "Enter") {
              window.removeEventListener("keydown", handler);
              resolve(e.target.value);
            }
          };
          window.addEventListener("keydown", handler);
        });
        // Add answer to chat
        setMessages((msgs) => [
          ...msgs,
          { sender: "user", text: answer }
        ]);
        data.questions[field] = answer;
      }
      // Send filled info
      const res2 = await fetch(`${API_URL}/process_query/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: input, missing_info: data.questions })
      });
      const data2 = await res2.json();
      setEnhancedPrompt(data2.enhanced_prompt);
      setMessages((msgs) => [
        ...msgs,
        { sender: "system", text: "ලිපිය ජනනය කරමින්..." }
      ]);
      await generateLetter(input, data2.enhanced_prompt);
    } else if (data.status === "complete") {
      setEnhancedPrompt(data.enhanced_prompt);
      setMessages((msgs) => [
        ...msgs,
        { sender: "system", text: "ලිපිය ජනනය කරමින්..." }
      ]);
      await generateLetter(input, data.enhanced_prompt);
    }
    setWaiting(false);
    setInput("");
  };

  const generateLetter = async (original, enhanced) => {
    const res = await fetch(`${API_URL}/generate_letter/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ original_prompt: original, enhanced_prompt: enhanced })
    });
    const data = await res.json();
    setLetter(data.generated_letter);
    setMessages((msgs) => [
      ...msgs,
      { sender: "system", text: "👇 ජනනය වූ ලිපිය පහතින්:" }
    ]);
  };

  const handleInput = (e) => setInput(e.target.value);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !waiting) sendMessage();
  };

  return (
    <div style={{ maxWidth: 600, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h2>සිංහල ලිපි සංස්කරණය</h2>
      <div style={{
        border: "1px solid #ccc", borderRadius: 8, padding: 16, minHeight: 300, background: "#fafafa"
      }}>
        {messages.map((msg, i) => (
          <div key={i} style={{
            textAlign: msg.sender === "user" ? "right" : "left",
            margin: "8px 0"
          }}>
            <span style={{
              display: "inline-block",
              background: msg.sender === "user" ? "#e0f7fa" : "#f1f8e9",
              padding: "8px 12px",
              borderRadius: 16,
              maxWidth: "80%",
              wordBreak: "break-word"
            }}>
              {msg.text}
            </span>
          </div>
        ))}
        {waiting && <div>⏳ කරුණාකර රැඳී සිටින්න...</div>}
      </div>
      <div style={{ marginTop: 16, display: "flex" }}>
        <input
          type="text"
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          disabled={waiting}
          placeholder="ඔබගේ ලිපි ඉල්ලීම මෙහි ලියන්න..."
          style={{ flex: 1, padding: 10, borderRadius: 8, border: "1px solid #ccc" }}
        />
        <button
          onClick={sendMessage}
          disabled={waiting}
          style={{
            marginLeft: 8, padding: "10px 18px", borderRadius: 8, background: "#1976d2", color: "#fff", border: "none"
          }}
        >
          යවන්න
        </button>
      </div>
      {letter && (
        <div style={{
          marginTop: 32,
          background: "#fffde7",
          border: "1px solid #ffe082",
          borderRadius: 8,
          padding: 20
        }}>
          <h3>📝 ජනනය වූ ලිපිය</h3>
          <pre style={{ whiteSpace: "pre-wrap", fontFamily: "inherit" }}>{letter}</pre>
        </div>
      )}
    </div>
  );
}

export default Chat;
