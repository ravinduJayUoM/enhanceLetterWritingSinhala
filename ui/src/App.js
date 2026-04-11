import React, { useState } from "react";
import { isLoggedIn, getProfile, logout } from "./auth";
import Login from "./Login";
import Register from "./Register";
import LetterChat from "./LetterChat";
import Admin from "./Admin";

function App() {
  // Serve admin dashboard for /admin path
  if (window.location.pathname.startsWith("/admin")) {
    return <Admin />;
  }

  const [page, setPage] = useState(isLoggedIn() ? "chat" : "login");
  const [profile, setProfile] = useState(getProfile());

  const handleLogin = (userProfile) => {
    setProfile(userProfile);
    setPage("chat");
  };

  const handleLogout = () => {
    logout();
    setProfile(null);
    setPage("login");
  };

  if (page === "login") {
    return <Login onLogin={handleLogin} onGoRegister={() => setPage("register")} />;
  }

  if (page === "register") {
    return <Register onGoLogin={() => setPage("login")} />;
  }

  return <LetterChat profile={profile} onLogout={handleLogout} />;
}

export default App;
