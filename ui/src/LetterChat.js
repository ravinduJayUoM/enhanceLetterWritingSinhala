import React, { useState } from "react";
import MessageList from "./components/MessageList";
import InputBar from "./components/InputBar";
import LetterDisplay from "./components/LetterDisplay";
import GapForm from "./components/GapForm";
import { getToken } from "./auth";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Stages of the conversation
// ---------------------------------------------------------------------------
// "idle"       — waiting for the initial prompt
// "gap_form"   — showing a form with all missing fields at once
// "generating" — waiting for the letter generation API call
// "done"       — letter is ready; user can start over
// ---------------------------------------------------------------------------

const WELCOME = "ලිපිය ලිවීම සඳහා ඔබගේ ඉල්ලීම සිංහලෙන් ඇතුළත් කරන්න.";

export default function LetterChat({ profile, onLogout }) {
  const [messages, setMessages] = useState([
    { sender: "system", text: WELCOME },
  ]);
  const [input, setInput] = useState("");
  const [waiting, setWaiting] = useState(false);
  const [letter, setLetter] = useState(null);

  const [stage, setStage] = useState("idle");
  const [gapFields, setGapFields] = useState([]);      // [{field, question}] for the form
  const [autoFilled, setAutoFilled] = useState({});    // pre-filled answers (e.g. sender)
  const [originalPrompt, setOriginalPrompt] = useState("");
  const [letterCategory, setLetterCategory] = useState("general");
  const [ratingStatus, setRatingStatus] = useState(null);

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  const addMessage = (sender, text) =>
    setMessages((prev) => [...prev, { sender, text }]);

  const reset = () => {
    setMessages([{ sender: "system", text: WELCOME }]);
    setInput("");
    setWaiting(false);
    setLetter(null);
    setStage("idle");
    setGapFields([]);
    setAutoFilled({});
    setOriginalPrompt("");
    setLetterCategory("general");
    setRatingStatus(null);
  };

  // ------------------------------------------------------------------
  // Step: generate the letter once we have all info
  // ------------------------------------------------------------------
  const generateLetter = async (enhancedPrompt) => {
    setStage("generating");
    addMessage("system", "ලිපිය ජනනය කරමින්… මොහොතක් රැඳී සිටින්න.");
    setWaiting(true);

    const senderInfo = profile ? {
      full_name: profile.full_name || "",
      title: profile.title || "",
      address_line1: profile.address_line1 || "",
      address_line2: profile.address_line2 || "",
      phone: profile.phone || "",
    } : null;

    try {
      const res = await fetch(`${API_URL}/generate_letter/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${getToken()}`,
        },
        body: JSON.stringify({ enhanced_prompt: enhancedPrompt, sender_info: senderInfo }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setLetter(data.generated_letter);
      addMessage("system", "ලිපිය සූදානම් ✓ — පහතින් බලන්න.");
      setStage("done");
    } catch (err) {
      addMessage("system", `දෝෂයකි: ${err.message}`);
      setStage("idle");
    } finally {
      setWaiting(false);
      setInput("");
    }
  };

  // ------------------------------------------------------------------
  // Step: process the query (or re-process after gap form submitted)
  // ------------------------------------------------------------------
  const processQuery = async (prompt, missingInfo = null) => {
    setWaiting(true);
    try {
      const body = { prompt };
      if (missingInfo) body.missing_info = missingInfo;

      const res = await fetch(`${API_URL}/process_query/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${getToken()}`,
        },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();

      if (data.status === "incomplete") {
        let fields = Object.entries(data.questions).map(([field, question]) => ({
          field,
          question,
        }));

        // Auto-fill sender from logged-in profile — never ask the user for it
        const prefilled = {};
        if (profile?.full_name) {
          if (fields.some((f) => f.field === "sender")) {
            prefilled["sender"] = profile.full_name;
            fields = fields.filter((f) => f.field !== "sender");
          }
        }

        setLetterCategory(data.extracted_info?.letter_type || "general");

        if (fields.length === 0) {
          // All gaps auto-filled — go straight to generation
          await processQuery(prompt, prefilled);
          return;
        }

        // Show all remaining questions as a single form
        setAutoFilled(prefilled);
        setGapFields(fields);
        setStage("gap_form");
        setWaiting(false);
      } else {
        // All info present — go straight to generation
        setLetterCategory(data.extracted_info?.letter_type || "general");
        setRatingStatus(null);
        await generateLetter(data.enhanced_prompt);
      }
    } catch (err) {
      addMessage("system", `දෝෂයකි: ${err.message}`);
      setStage("idle");
      setWaiting(false);
    }
  };

  // ------------------------------------------------------------------
  // Gap form submitted — merge answers and re-process
  // ------------------------------------------------------------------
  const handleGapFormSubmit = async (answers) => {
    const allAnswers = { ...autoFilled, ...answers };
    setStage("generating");
    await processQuery(originalPrompt, allAnswers);
  };

  // ------------------------------------------------------------------
  // Rate the generated letter
  // ------------------------------------------------------------------
  const handleRate = async (stars) => {
    setRatingStatus("saving");
    try {
      const res = await fetch(`${API_URL}/rate_letter/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          letter_content: letter,
          rating: stars,
          original_prompt: originalPrompt,
          letter_category: letterCategory,
        }),
      });
      const data = await res.json();
      setRatingStatus(data.added_to_index ? "indexed" : "saved");
    } catch {
      setRatingStatus("error");
    }
  };

  // ------------------------------------------------------------------
  // Main send handler — only active in idle/done stage
  // ------------------------------------------------------------------
  const handleSend = async () => {
    const text = input.trim();
    if (!text || waiting) return;

    addMessage("user", text);
    setInput("");

    if (stage === "done") reset();
    setOriginalPrompt(text);
    await processQuery(text);
  };

  const placeholder =
    stage === "idle" || stage === "done"
      ? "ලිපි ඉල්ලීම සිංහලෙන් ඇතුළත් කරන්න…"
      : "";

  return (
    <div
      style={{
        maxWidth: 680,
        margin: "36px auto",
        fontFamily: "'Segoe UI', 'Noto Sans Sinhala', sans-serif",
        padding: "0 16px",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
        <h2 style={{ color: "#1a237e", margin: 0 }}>SinhalaLipi</h2>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {profile && (
            <span style={{ fontSize: 13, color: "#555" }}>
              {profile.full_name}
            </span>
          )}
          <button
            onClick={onLogout}
            style={{
              fontSize: 13, padding: "5px 14px", borderRadius: 6,
              border: "1px solid #ccc", background: "#fff",
              cursor: "pointer", color: "#555",
            }}
          >
            Logout
          </button>
        </div>
      </div>

      <MessageList messages={messages} waiting={waiting} />

      {stage === "gap_form" && (
        <GapForm fields={gapFields} onSubmit={handleGapFormSubmit} />
      )}

      {(stage === "idle" || stage === "done") && (
        <>
          <InputBar
            value={input}
            onChange={setInput}
            onSend={handleSend}
            disabled={waiting}
            placeholder={placeholder}
          />
          {stage === "done" && (
            <p style={{ textAlign: "center", marginTop: 12, color: "#555", fontSize: 13 }}>
              නව ලිපියක් ලිවීමට ඉහළ කොටුවේ නැවත ඇතුළත් කරන්න.
            </p>
          )}
        </>
      )}

      <LetterDisplay
        letter={letter}
        onRate={handleRate}
        ratingStatus={ratingStatus}
      />
    </div>
  );
}
