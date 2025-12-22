"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

interface Organization {
  id: string;
  name: string;
  members: string[];
  projects: string[];
}

export default function OrganizationsPage() {
  const router = useRouter();
  const [userId, setUserId] = useState<string | null>(null);
  const [username, setUsername] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [isFetchingOrgs, setIsFetchingOrgs] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string>("");
  const [showCreateForm, setShowCreateForm] = useState(false);

  // Create form fields
  const [orgName, setOrgName] = useState<string>("");
  const [orgMembers, setOrgMembers] = useState<string[]>([]);
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
      fetchOrganizations(storedUserId);
      setIsLoading(false);
    }
  }, [router]);

  const fetchOrganizations = async (user_id: string) => {
    setIsFetchingOrgs(true);
    try {
      const requestBody = {
        user_id: user_id,
        organization_id: "",
        force_refresh: false,
      };

      const response = await fetch(
        `http://localhost:8000/organizations/list_orgs`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        }
      );
      const data = await response.json();

      if (response.ok && Array.isArray(data)) {
        setOrganizations(data);
      } else {
        console.error("Failed to fetch organizations:", data);
      }
    } catch (err) {
      console.error("Error fetching organizations:", err);
    } finally {
      setIsFetchingOrgs(false);
    }
  };

  const handleAddMember = () => {
    if (!newMemberEmail.trim() || !newMemberUsername.trim()) {
      setError("Please provide both email and username for the member");
      return;
    }

    if (orgMembers.includes(newMemberUsername)) {
      setError("Member already added");
      return;
    }

    setOrgMembers([...orgMembers, newMemberUsername]);
    setNewMemberEmail("");
    setNewMemberUsername("");
    setError("");
  };

  const handleRemoveMember = (username: string) => {
    setOrgMembers(orgMembers.filter((m) => m !== username));
  };

  const handleCreateOrganization = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsCreating(true);

    if (!orgName.trim()) {
      setError("Please provide an organization name");
      setIsCreating(false);
      return;
    }

    if (!userId) {
      setError("User ID not found. Please log in again.");
      setIsCreating(false);
      return;
    }

    try {
      const requestBody = {
        id: "",
        name: orgName,
        members: orgMembers,
        projects: [],
      };

      const response = await fetch(
        `http://localhost:8000/organizations/create_org?user_id=${encodeURIComponent(userId)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        }
      );

      const data = await response.json();

      if (response.ok) {
        setShowCreateForm(false);
        setOrgName("");
        setOrgMembers([]);
        await fetchOrganizations(userId);
      } else {
        setError(data.detail || "Failed to create organization");
      }
    } catch (err) {
      setError("Error connecting to server. Please ensure the API is running.");
    } finally {
      setIsCreating(false);
    }
  };

  const handleDeleteOrganization = async (orgId: string, orgName: string) => {
    if (!confirm(`Are you sure you want to delete "${orgName}"?`)) {
      return;
    }

    if (!userId) return;

    try {
      const requestBody = {
        user_id: userId,
        organization_id: orgId,
        force_refresh: false,
      };

      const response = await fetch(
        `http://localhost:8000/organizations/delete_org`,
        {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        }
      );

      if (response.ok) {
        await fetchOrganizations(userId);
      } else {
        const data = await response.json();
        alert("Failed to delete organization: " + (data.detail || "Unknown error"));
      }
    } catch (err) {
      alert("Error connecting to server. Please ensure the API is running.");
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

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Organizations</h1>
            <p className="text-white/70">
              Manage your organizations and collaborate with teams
            </p>
          </div>
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="px-6 py-3 bg-white hover:bg-gray-200 text-black font-semibold rounded-lg transition-all flex items-center gap-2"
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
            {showCreateForm ? "Cancel" : "Create Organization"}
          </button>
        </div>

        {/* Create Organization Form */}
        {showCreateForm && (
          <div className="bg-black/50 border border-white/20 rounded-lg p-8 backdrop-blur-sm mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">Create New Organization</h2>

            <form onSubmit={handleCreateOrganization} className="space-y-6">
              {/* Organization Name */}
              <div>
                <label className="block text-white/90 text-sm font-medium mb-2">
                  Organization Name <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  value={orgName}
                  onChange={(e) => setOrgName(e.target.value)}
                  className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:border-white/40 focus:bg-white/10 transition-all"
                  placeholder="Enter organization name"
                  required
                />
              </div>

              {/* Add Members */}
              <div>
                <label className="block text-white/90 text-sm font-medium mb-3">
                  Members (Optional)
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
                {orgMembers.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-white/70 text-sm mb-2">
                      {orgMembers.length} member{orgMembers.length !== 1 ? "s" : ""} added
                    </p>
                    {orgMembers.map((member, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-3 bg-white/5 border border-white/20 rounded-lg"
                      >
                        <div className="text-white font-medium">{member}</div>
                        <button
                          type="button"
                          onClick={() => handleRemoveMember(member)}
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

              <button
                type="submit"
                disabled={isCreating}
                className="w-full py-3 bg-white hover:bg-gray-200 disabled:bg-gray-400 text-black font-semibold rounded-lg transition-all"
              >
                {isCreating ? "Creating Organization..." : "Create Organization"}
              </button>
            </form>
          </div>
        )}

        {/* Organizations List */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {isFetchingOrgs && organizations.length === 0 ? (
            <div className="col-span-full text-center py-12">
              <div className="animate-spin w-12 h-12 border-4 border-white/20 border-t-white rounded-full mx-auto mb-4"></div>
              <p className="text-white/70">Loading organizations...</p>
            </div>
          ) : organizations.length === 0 ? (
            <div className="col-span-full text-center py-12">
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
                  d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                />
              </svg>
              <p className="text-white/70 text-lg">No organizations yet</p>
              <p className="text-white/50 text-sm">Create your first organization to get started</p>
            </div>
          ) : (
            organizations.map((org) => (
              <div
                key={org.id}
                className="bg-black/50 border border-white/20 rounded-lg p-6 backdrop-blur-sm hover:border-white/30 transition-all"
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-xl font-bold text-white">{org.name}</h3>
                  <button
                    onClick={() => handleDeleteOrganization(org.id, org.name)}
                    className="p-2 hover:bg-red-500/20 text-red-400 rounded-lg transition-all"
                    title="Delete organization"
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
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                  </button>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-white/70">
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
                        d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"
                      />
                    </svg>
                    <span className="text-sm">
                      {org.members.length} member{org.members.length !== 1 ? "s" : ""}
                    </span>
                  </div>

                  <div className="flex items-center gap-2 text-white/70">
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
                        d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                      />
                    </svg>
                    <span className="text-sm">
                      {org.projects.length} project{org.projects.length !== 1 ? "s" : ""}
                    </span>
                  </div>

                  {org.members.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-white/10">
                      <p className="text-white/50 text-xs mb-2">Members:</p>
                      <div className="flex flex-wrap gap-2">
                        {org.members.slice(0, 3).map((member, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-1 bg-white/10 text-white/80 text-xs rounded border border-white/20"
                          >
                            {member}
                          </span>
                        ))}
                        {org.members.length > 3 && (
                          <span className="px-2 py-1 bg-white/10 text-white/80 text-xs rounded border border-white/20">
                            +{org.members.length - 3} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
