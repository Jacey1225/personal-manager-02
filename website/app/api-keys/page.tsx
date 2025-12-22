"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

interface TokenData {
  name: string;
  token: string;
}

export default function APIKeysPage() {
  const router = useRouter();
  const [userId, setUserId] = useState<string | null>(null);
  const [username, setUsername] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [tokenName, setTokenName] = useState<string>("");
  const [selectedScopes, setSelectedScopes] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [isFetchingTokens, setIsFetchingTokens] = useState(false);
  const [showTokenModal, setShowTokenModal] = useState(false);
  const [newToken, setNewToken] = useState<string | null>(null);
  const [tokenCopied, setTokenCopied] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string>("");
  const [tokens, setTokens] = useState<TokenData[]>([]);
  const [revokingToken, setRevokingToken] = useState<string | null>(null);

  const availableScopes = [
    { value: "widgets:read", label: "Read Widgets", description: "View and retrieve widget data" },
    { value: "widgets:write", label: "Write Widgets", description: "Create and modify widgets" },
    { value: "projects:read", label: "Read Projects", description: "View project information" },
    { value: "projects:write", label: "Write Projects", description: "Create and modify projects" },
  ];

  useEffect(() => {
    const storedUserId = localStorage.getItem("user_id");
    const storedUsername = localStorage.getItem("username");
    if (!storedUserId) {
      router.push("/login");
    } else {
      setUserId(storedUserId);
      if (storedUsername) {
        setUsername(storedUsername);
        fetchUserTokens(storedUsername);
      }
      setIsLoading(false);
    }
  }, [router]);

  const fetchUserTokens = async (user: string) => {
    setIsFetchingTokens(true);
    try {
      const response = await fetch(
        `http://localhost:8000/oauth/list_tokens?username=${encodeURIComponent(user)}`
      );
      const data = await response.json();

      if (response.ok && data.status === "success") {
        setTokens(data.tokens || []);
      } else {
        console.error("Failed to fetch tokens:", data);
      }
    } catch (err) {
      console.error("Error fetching tokens:", err);
    } finally {
      setIsFetchingTokens(false);
    }
  };

  const handleCreateToken = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsCreating(true);

    if (!tokenName.trim()) {
      setError("Please provide a name for your token");
      setIsCreating(false);
      return;
    }

    if (selectedScopes.length === 0) {
      setError("Please select at least one scope");
      setIsCreating(false);
      return;
    }

    try {
      // Create form data for OAuth2PasswordRequestForm
      const formData = new URLSearchParams();
      formData.append("username", username);
      formData.append("password", password);
      selectedScopes.forEach((scope) => {
        formData.append("scope", scope);
      });

      const response = await fetch(
        `http://localhost:8000/oauth/token?token_name=${encodeURIComponent(tokenName)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
          body: formData.toString(),
        }
      );

      const data = await response.json();

      if (response.ok) {
        setNewToken(data.access_token);
        setShowTokenModal(true);
        setPassword("");
        setTokenName("");
        setSelectedScopes([]);
        
        // Refresh the token list
        await fetchUserTokens(username);
      } else {
        setError(data.detail || "Failed to create token");
      }
    } catch (err) {
      setError("Error connecting to server. Please ensure the API is running.");
    } finally {
      setIsCreating(false);
    }
  };

  const handleRevokeToken = async (token: string, tokenName: string) => {
    if (!confirm(`Are you sure you want to revoke the token "${tokenName}"?`)) {
      return;
    }

    setRevokingToken(tokenName);
    try {
      const response = await fetch(
        `http://localhost:8000/oauth/revoke?username=${encodeURIComponent(username)}`,
        {
          method: "DELETE",
          headers: {
            "Authorization": `Bearer ${token}`,
          },
        }
      );

      const data = await response.json();

      if (response.ok && data.status === "success") {
        // Refresh the token list
        await fetchUserTokens(username);
      } else {
        alert("Failed to revoke token: " + (data.detail || "Unknown error"));
      }
    } catch (err) {
      alert("Error connecting to server. Please ensure the API is running.");
    } finally {
      setRevokingToken(null);
    }
  };

  const handleCopyToken = () => {
    if (newToken) {
      navigator.clipboard.writeText(newToken);
      setTokenCopied(true);
      setTimeout(() => setTokenCopied(false), 2000);
    }
  };

  const handleCloseModal = () => {
    setShowTokenModal(false);
    setNewToken(null);
    setTokenCopied(false);
  };

  const handleLogout = () => {
    localStorage.removeItem("user_id");
    localStorage.removeItem("username");
    router.push("/");
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black">
      {/* Top Navigation Bar */}
      <nav className="w-full border-b border-white/10 bg-black/50 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <Link
              href="/dashboard"
              className="text-2xl font-bold text-white hover:text-white/80 transition-colors"
            >
              NoManager
            </Link>

            <div className="flex items-center gap-6">
              <Link
                href="/dashboard"
                className="text-white/70 hover:text-white transition-colors"
              >
                Dashboard
              </Link>
              <span className="text-white/70 text-sm">
                User ID: {userId?.slice(0, 8)}...
              </span>
              <button
                onClick={handleLogout}
                className="px-4 py-2 border border-white/30 hover:border-white/50 text-white hover:bg-white/10 rounded-lg transition-all"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Developer API Keys</h1>
          <p className="text-white/70">
            Create and manage OAuth2 tokens for API access with custom scopes
          </p>
        </div>

        {/* Create Token Form */}
        <div className="bg-black/50 border border-white/20 rounded-lg p-8 backdrop-blur-sm mb-8">
          <h2 className="text-2xl font-bold text-white mb-6">Create New Token</h2>

          <form onSubmit={handleCreateToken} className="space-y-6">
            {/* Token Name Field */}
            <div>
              <label className="block text-white/90 text-sm font-medium mb-2">
                Token Name
              </label>
              <input
                type="text"
                value={tokenName}
                onChange={(e) => setTokenName(e.target.value)}
                className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:border-white/40 focus:bg-white/10 transition-all"
                placeholder="e.g., Production API, Development Token"
                required
              />
            </div>

            {/* Username Field */}
            <div>
              <label className="block text-white/90 text-sm font-medium mb-2">
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:border-white/40 focus:bg-white/10 transition-all"
                placeholder="Enter your username"
                required
              />
            </div>

            {/* Password Field */}
            <div>
              <label className="block text-white/90 text-sm font-medium mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:border-white/40 focus:bg-white/10 transition-all"
                  placeholder="Enter your password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-white/50 hover:text-white/80"
                >
                  {showPassword ? (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            {/* Scopes Selection */}
            <div>
              <label className="block text-white/90 text-sm font-medium mb-3">
                Select Scopes
              </label>
              <div className="space-y-3">
                {availableScopes.map((scope) => (
                  <label
                    key={scope.value}
                    className="flex items-start gap-3 p-4 bg-white/5 border border-white/20 rounded-lg cursor-pointer hover:bg-white/10 transition-all"
                  >
                    <input
                      type="checkbox"
                      checked={selectedScopes.includes(scope.value)}
                      onChange={() => handleScopeToggle(scope.value)}
                      className="mt-1 w-4 h-4 rounded border-white/30 bg-white/10 text-white focus:ring-2 focus:ring-white/50"
                    />
                    <div className="flex-1">
                      <div className="text-white font-medium">{scope.label}</div>
                      <div className="text-white/60 text-sm">{scope.description}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {error && (
              <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}

            <button
              type="submit"
              disabled={isCreating}
              className="w-full py-3 bg-white hover:bg-gray-200 disabled:bg-gray-400 text-black font-semibold rounded-lg transition-all"
            >
              {isCreating ? "Creating Token..." : "Create Token"}
            </button>
          </form>
        </div>

        {/* Existing Tokens List */}
        <div className="bg-black/50 border border-white/20 rounded-lg p-8 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-white">Your API Tokens</h2>
            <button
              onClick={() => fetchUserTokens(username)}
              disabled={isFetchingTokens}
              className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white text-sm rounded-lg transition-all disabled:opacity-50"
            >
              {isFetchingTokens ? "Refreshing..." : "Refresh"}
            </button>
          </div>

          {isFetchingTokens && tokens.length === 0 ? (
            <div className="text-center py-12">
              <div className="animate-spin w-12 h-12 border-4 border-white/20 border-t-white rounded-full mx-auto mb-4"></div>
              <p className="text-white/70">Loading tokens...</p>
            </div>
          ) : tokens.length === 0 ? (
            <div className="text-center py-12">
              <svg
                className="w-16 h-16 text-white/30 mx-auto mb-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"
                />
              </svg>
              <p className="text-white/70 text-lg">No API tokens yet</p>
              <p className="text-white/50 text-sm">Create your first token to get started</p>
            </div>
          ) : (
            <div className="space-y-4">
              {tokens.map((token, index) => (
                <div
                  key={index}
                  className="p-5 bg-white/5 border border-white/20 rounded-lg"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="text-white font-semibold text-lg">{token.name}</h3>
                        <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded border border-green-500/30">
                          Active
                        </span>
                      </div>
                      <div className="text-white/50 font-mono text-sm">
                        Token Hash: {token.token.slice(0, 30)}...
                      </div>
                    </div>
                    <button
                      onClick={() => handleRevokeToken(token.token, token.name)}
                      disabled={revokingToken === token.name}
                      className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 hover:border-red-500/50 text-red-400 hover:text-red-300 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      {revokingToken === token.name ? (
                        <>
                          <div className="animate-spin w-4 h-4 border-2 border-red-400/20 border-t-red-400 rounded-full"></div>
                          Revoking...
                        </>
                      ) : (
                        <>
                          <svg
                            className="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                            />
                          </svg>
                          Revoke
                        </>
                      )}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Token Modal */}
      {showTokenModal && newToken && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-black border border-white/30 rounded-lg max-w-2xl w-full p-8">
            <div className="mb-6">
              <div className="flex items-center gap-3 mb-2">
                <svg
                  className="w-8 h-8 text-green-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <h2 className="text-2xl font-bold text-white">Token Created Successfully!</h2>
              </div>
              <p className="text-white/70">
                Copy this token now. You won&apos;t be able to see it again after closing this modal.
              </p>
            </div>

            <div className="bg-white/5 border border-white/20 rounded-lg p-4 mb-6">
              <div className="flex items-center gap-3">
                <code className="flex-1 text-white font-mono text-sm break-all">
                  {newToken}
                </code>
                <button
                  onClick={handleCopyToken}
                  className="flex-shrink-0 p-2 bg-white/10 hover:bg-white/20 text-white rounded transition-all"
                  title="Copy token"
                >
                  {tokenCopied ? (
                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 mb-6">
              <div className="flex items-start gap-3">
                <svg
                  className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
                <div>
                  <p className="text-yellow-400 font-semibold mb-1">Important Security Notice</p>
                  <p className="text-yellow-400/80 text-sm">
                    Store this token securely. It provides access to your account with the selected scopes.
                  </p>
                </div>
              </div>
            </div>

            <button
              onClick={handleCloseModal}
              className="w-full py-3 bg-white hover:bg-gray-200 text-black font-semibold rounded-lg transition-all"
            >
              I&apos;ve Copied My Token
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
