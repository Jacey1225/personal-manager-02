"use client";

import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    username: "",
    password: "",
  });
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      // Call your API login endpoint
      const response = await fetch(`http://localhost:8000/auth/login?username=${encodeURIComponent(formData.username)}&password=${encodeURIComponent(formData.password)}`, {
        method: "GET",
      });

      const data = await response.json();

      if (data.status === "success") {
        // Store user_id and username in localStorage
        localStorage.setItem("user_id", data.user_id);
        localStorage.setItem("username", formData.username);
        // Redirect to dashboard
        router.push("/dashboard");
      } else {
        setError(data.message || "Login failed. Please check your credentials.");
      }
    } catch (err) {
      setError("Failed to connect to server. Please try again.");
      console.error("Login error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black">
      {/* Navigation Bar */}
      <nav className="w-full border-b border-white/10">
        <div className="backdrop-blur-md bg-black/50">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <Link 
                href="/" 
                className="text-2xl font-bold text-white hover:text-white/80 transition-colors"
              >
                NoManager
              </Link>

              <div className="flex items-center gap-8">
                <Link
                  href="/projects"
                  className="text-white/80 hover:text-white font-medium transition-colors"
                >
                  Projects
                </Link>
                <Link
                  href="/signup"
                  className="px-4 py-2 bg-white hover:bg-gray-200 text-black font-medium rounded-lg transition-all"
                >
                  Sign Up
                </Link>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Login Form */}
      <div className="flex items-center justify-center px-6 py-16">
        <div className="w-full max-w-md">
          {/* Form Card */}
          <div className="backdrop-blur-xl bg-black/80 border border-white/30 rounded-2xl shadow-2xl p-8">
            <div className="text-center mb-8">
              <h1 className="text-4xl font-bold text-white mb-2">
                Welcome Back
              </h1>
              <p className="text-white/70">
                Sign in to your NoManager account
              </p>
            </div>

            {error && (
              <div className="mb-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg">
                <p className="text-red-200 text-sm">{error}</p>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Username */}
              <div>
                <label htmlFor="username" className="block text-sm font-medium text-white/90 mb-2">
                  Username
                </label>
                <input
                  type="text"
                  id="username"
                  name="username"
                  value={formData.username}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="Enter your username"
                />
              </div>

              {/* Password */}
              <div>
                <label htmlFor="password" className="block text-sm font-medium text-white/90 mb-2">
                  Password
                </label>
                <input
                  type="password"
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="Enter your password"
                />
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full py-3 px-4 bg-white hover:bg-gray-200 disabled:bg-gray-400 text-black font-semibold rounded-lg shadow-lg transition-all duration-200 hover:scale-105 disabled:hover:scale-100 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Signing in...
                  </span>
                ) : (
                  "Sign In"
                )}
              </button>
            </form>

            {/* Signup Link */}
            <div className="mt-6 text-center">
              <p className="text-white/70 text-sm">
                Don't have an account?{" "}
                <Link href="/signup" className="text-white hover:text-gray-300 font-medium transition-colors">
                  Create one now
                </Link>
              </p>
            </div>
          </div>

          {/* Back to Home */}
          <div className="mt-6 text-center">
            <Link
              href="/"
              className="inline-flex items-center text-white/60 hover:text-white/90 transition-colors"
            >
              <svg
                className="mr-2 w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M10 19l-7-7m0 0l7-7m-7 7h18"
                />
              </svg>
              Back to home
            </Link>
          </div>
        </div>
      </div>
    </main>
  );
}
