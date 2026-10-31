export interface StoredUser {
  name?: string;
  email: string;
  phone?: string;
}

const USER_KEY = "dd_user";

export function readStoredUser(): StoredUser | null {
  try {
    const raw = window.localStorage.getItem(USER_KEY);
    if (!raw) return null;
    const user = JSON.parse(raw) as Partial<StoredUser>;
    return typeof user.email === "string" && user.email.trim()
      ? { ...user, email: user.email.trim().toLowerCase() }
      : null;
  } catch {
    return null;
  }
}

export function saveStoredUser(user: StoredUser): void {
  const normalizedUser = { ...user, email: user.email.trim().toLowerCase() };
  window.localStorage.setItem(USER_KEY, JSON.stringify(normalizedUser));
  if (readStoredUser()?.email !== normalizedUser.email) {
    throw new Error("Login session could not be saved on this device.");
  }
}

export function clearStoredUser(): void {
  window.localStorage.removeItem(USER_KEY);
}
