"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

interface ProjectMember {
  email: string;
  username: string;
}

export default function CreateProjectPage() {
  const router = useRouter();
  const [userId, setUserId] = useState<string | null>(null);
  const [username, setUsername] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string>("");
  
  // Form fields
  const [projectName, setProjectName] = useState<string>("");
  const [projectTransparency, setProjectTransparency] = useState<boolean>(true);
  const [projectMembers, setProjectMembers] = useState<ProjectMember[]>([]);
  const [newMemberEmail, setNewMemberEmail] = useState<string>("");
  const [newMemberUsername, setNewMemberUsername] = useState<string>("");

  useEffect(() => {
    const storedUserId = localStorage.getItem("user_id");
    const storedUsername = localStorage.getItem("username");
    if (!storedUserId) {
      router.push("/login");
    } else {
      setUserId(storedUserId);
      if (storedUsername) {
        setUsername(storedUsername);
      }
      setIsLoading(false);
    }
  }, [router]);

  const handleAddMember = () => {
    if (!newMemberEmail.trim() || !newMemberUsername.trim()) {
      setError("Please provide both email and username for the member");
      return;
    }

    // Check for duplicates
    if (projectMembers.some(m => m.email === newMemberEmail || m.username === newMemberUsername)) {
      setError("Member with this email or username already added");
      return;
    }

    setProjectMembers([...projectMembers, { email: newMemberEmail, username: newMemberUsername }]);
    setNewMemberEmail("");
    setNewMemberUsername("");
    setError("");
  };

  const handleRemoveMember = (index: number) => {
    setProjectMembers(projectMembers.filter((_, i) => i !== index));
  };

  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsCreating(true);

    if (!projectName.trim()) {
      setError("Please provide a project name");
      setIsCreating(false);
      return;
    }

    if (!userId) {
      setError("User ID not found. Please log in again.");
      setIsCreating(false);
      return;
    }

    try {
      // Format project_members as list of tuples (email, username)
      const formattedMembers = projectMembers.map(m => [m.email, m.username]);

      const requestBody = {
        project_name: projectName,
        project_transparency: projectTransparency,
        project_likes: 0,
        project_members: formattedMembers,
        user_id: userId,
      };

      const response = await fetch("http://localhost:8000/projects/create_project", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();

      if (response.ok) {
        // Redirect to dashboard or projects list
        router.push("/dashboard");
      } else {
        setError(data.detail || "Failed to create project");
      }
    } catch (err) {
      setError("Error connecting to server. Please ensure the API is running.");
    } finally {
      setIsCreating(false);
    }
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

      <div className="max-w-4xl mx-auto px-6 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Create New Project</h1>
          <p className="text-white/70">
            Set up a new project and invite team members to collaborate
          </p>
        </div>

        <div className="bg-black/50 border border-white/20 rounded-lg p-8 backdrop-blur-sm">
          <form onSubmit={handleCreateProject} className="space-y-6">
            {/* Project Name */}
            <div>
              <label className="block text-white/90 text-sm font-medium mb-2">
                Project Name <span className="text-red-400">*</span>
              </label>
              <input
                type="text"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:border-white/40 focus:bg-white/10 transition-all"
                placeholder="Enter project name"
                required
              />
            </div>

            {/* Project Transparency */}
            <div>
              <label className="block text-white/90 text-sm font-medium mb-3">
                Project Visibility
              </label>
              <div className="space-y-3">
                <label className="flex items-start gap-3 p-4 bg-white/5 border border-white/20 rounded-lg cursor-pointer hover:bg-white/10 transition-all">
                  <input
                    type="radio"
                    name="transparency"
                    checked={projectTransparency === true}
                    onChange={() => setProjectTransparency(true)}
                    className="mt-1 w-4 h-4"
                  />
                  <div className="flex-1">
                    <div className="text-white font-medium">Public</div>
                    <div className="text-white/60 text-sm">
                      Anyone can view this project and its contents
                    </div>
                  </div>
                </label>

                <label className="flex items-start gap-3 p-4 bg-white/5 border border-white/20 rounded-lg cursor-pointer hover:bg-white/10 transition-all">
                  <input
                    type="radio"
                    name="transparency"
                    checked={projectTransparency === false}
                    onChange={() => setProjectTransparency(false)}
                    className="mt-1 w-4 h-4"
                  />
                  <div className="flex-1">
                    <div className="text-white font-medium">Private</div>
                    <div className="text-white/60 text-sm">
                      Only project members can view and access this project
                    </div>
                  </div>
                </label>
              </div>
            </div>

            {/* Add Project Members */}
            <div>
              <label className="block text-white/90 text-sm font-medium mb-3">
                Project Members (Optional)
              </label>
              
              <div className="space-y-3 mb-4">
                <div className="grid grid-cols-2 gap-3">
                  <input
                    type="email"
                    value={newMemberEmail}
                    onChange={(e) => setNewMemberEmail(e.target.value)}
                    className="px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:border-white/40 focus:bg-white/10 transition-all"
                    placeholder="Member email"
                  />
                  <input
                    type="text"
                    value={newMemberUsername}
                    onChange={(e) => setNewMemberUsername(e.target.value)}
                    className="px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:border-white/40 focus:bg-white/10 transition-all"
                    placeholder="Member username"
                  />
                </div>
                <button
                  type="button"
                  onClick={handleAddMember}
                  className="w-full py-3 bg-white/10 hover:bg-white/20 border border-white/30 text-white rounded-lg transition-all flex items-center justify-center gap-2"
                >
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 4v16m8-8H4"
                    />
                  </svg>
                  Add Member
                </button>
              </div>

              {/* Members List */}
              {projectMembers.length > 0 && (
                <div className="space-y-2">
                  <p className="text-white/70 text-sm mb-2">
                    {projectMembers.length} member{projectMembers.length !== 1 ? 's' : ''} added
                  </p>
                  {projectMembers.map((member, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 bg-white/5 border border-white/20 rounded-lg"
                    >
                      <div>
                        <div className="text-white font-medium">{member.username}</div>
                        <div className="text-white/60 text-sm">{member.email}</div>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleRemoveMember(index)}
                        className="p-2 hover:bg-red-500/20 text-red-400 rounded-lg transition-all"
                      >
                        <svg
                          className="w-5 h-5"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M6 18L18 6M6 6l12 12"
                          />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {error && (
              <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}

            <div className="flex gap-4">
              <button
                type="button"
                onClick={() => router.back()}
                className="flex-1 py-3 bg-white/10 hover:bg-white/20 border border-white/30 text-white rounded-lg transition-all"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isCreating}
                className="flex-1 py-3 bg-white hover:bg-gray-200 disabled:bg-gray-400 text-black font-semibold rounded-lg transition-all"
              >
                {isCreating ? "Creating Project..." : "Create Project"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
