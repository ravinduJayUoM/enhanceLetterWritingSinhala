import React, { useState } from "react";
import MessageList from "./components/MessageList";
import InputBar from "./components/InputBar";
import LetterDisplay from "./components/LetterDisplay";

const API_URL = "http://34.87.52.138:8000";

// ---------------------------------------------------------------------------
// Stages of the conversation
// ---------------------------------------------------------------------------
// "idle"          — waiting for the initial prompt
// "collecting"    — asking gap-filling questions one by one
// "generating"    — waiting for the letter generation API call
// "done"          — letter is ready; user can start over
// ---------------------------------------------------------------------------

const WELCOME = "ලිපිය ලිවීම සඳහා ඔබගේ ඉල්ලීම සිංහලෙන් ඇතුළත් කරන්න.";

export default function LetterChat() {
  const [messages, setMessages] = useState([
    { sender: "system", text: WELCOME },
  ]);
  const [input, setInput] = useState("");
  const [waiting, setWaiting] = useState(false);
  const [letter, setLetter] = useState(null);

  // Gap-filling state
  const [stage, setStage] = useState("idle"); // idle | collecting | generating | done
  const [gapFields, setGapFields] = useState([]); // ordered field names still to ask
  const [gapAnswers, setGapAnswers] = useState({}); // { field: answer }
  const [originalPrompt, setOriginalPrompt] = useState("");
  const [letterCategory, setLetterCategory] = useState("general");
  const [ratingStatus, setRatingStatus] = useState(null); // null | saving | indexed | saved | error

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
    setGapAnswers({});
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
    try {
      const res = await fetch(`${API_URL}/generate_letter/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enhanced_prompt: enhancedPrompt }),
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
  // Step: process the query (or re-process after gap answers collected)
  // ------------------------------------------------------------------
  const processQuery = async (prompt, missingInfo = null) => {
    setWaiting(true);
    try {
      const body = { prompt };
      if (missingInfo) body.missing_info = missingInfo;

      const res = await fetch(`${API_URL}/process_query/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();

      if (data.status === "incomplete") {
        // Convert questions object to an ordered list: [{field, question}, ...]
        const fields = Object.entries(data.questions).map(([field, question]) => ({
          field,
          question,
        }));
        setGapFields(fields);
        setGapAnswers({});
        setStage("collecting");
        setLetterCategory(data.extracted_info?.letter_type || "general");
        // Ask the first question
        addMessage("system", fields[0].question);
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
  // Main send handler — behaviour depends on current stage
  // ------------------------------------------------------------------
  const handleSend = async () => {
    const text = input.trim();
    if (!text || waiting) return;

    addMessage("user", text);
    setInput("");

    if (stage === "idle" || stage === "done") {
      // Fresh query — start from scratch if done
      if (stage === "done") reset();
      setOriginalPrompt(text);
      await processQuery(text);
      return;
    }

    if (stage === "collecting") {
      const currentField = gapFields[0];
      const updatedAnswers = { ...gapAnswers, [currentField.field]: text };
      setGapAnswers(updatedAnswers);

      const remaining = gapFields.slice(1);
      setGapFields(remaining);

      if (remaining.length > 0) {
        // More questions to ask
        addMessage("system", remaining[0].question);
      } else {
        // All gap answers collected — re-process
        setWaiting(true);
        await processQuery(originalPrompt, updatedAnswers);
      }
    }
  };

  // ------------------------------------------------------------------
  // Input placeholder changes with stage
  // ------------------------------------------------------------------
  const placeholder =
    stage === "idle" || stage === "done"
      ? "ලිපි ඉල්ලීම සිංහලෙන් ඇතුළත් කරන්න…"
      : stage === "collecting"
      ? "පිළිතුර ඇතුළත් කරන්න…"
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
      <h2 style={{ textAlign: "center", color: "#1a237e", marginBottom: 20 }}>
        Sinhala Letters LK
      </h2>

      <MessageList messages={messages} waiting={waiting} />

      <InputBar
        value={input}
        onChange={setInput}
        onSend={handleSend}
        disabled={waiting || stage === "generating"}
        placeholder={placeholder}
      />

      {stage === "done" && (
        <p style={{ textAlign: "center", marginTop: 12, color: "#555", fontSize: 13 }}>
          නව ලිපියක් ලිවීමට ඉහළ කොටුවේ නැවත ඇතුළත් කරන්න.
        </p>
      )}

      <LetterDisplay
        letter={letter}
        onRate={handleRate}
        ratingStatus={ratingStatus}
      />
    </div>
  );
}
