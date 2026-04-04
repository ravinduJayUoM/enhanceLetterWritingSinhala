const TOKEN_KEY = "sinhala_rag_token";
const PROFILE_KEY = "sinhala_rag_profile";

export function saveToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function saveProfile(profile) {
  localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
}

export function getProfile() {
  const raw = localStorage.getItem(PROFILE_KEY);
  return raw ? JSON.parse(raw) : null;
}

export function logout() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(PROFILE_KEY);
}

export function isLoggedIn() {
  return !!getToken();
}

export function authHeaders() {
  return { Authorization: `Bearer ${getToken()}` };
}
