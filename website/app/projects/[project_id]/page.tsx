"use client";

import Link from "next/link";
import { useRouter, useParams } from "next/navigation";
import { useEffect, useState } from "react";

interface Project {
  project_id: string;
  project_name: string;
  project_transparency: boolean;
  project_likes: number;
  project_members: string[];
  widgets: string[];
  organizations: string[];
}

interface Widget {
  widget_id: string;
  name: string;
  size: "small" | "medium" | "large" | "extra_large";
  interactions: {
    [endpoint: string]: {
      params: Record<string, any>;
      headers: Record<string, string>;
      refresh_interval: number;
      components: WidgetComponent[];
      logic: string;
    };
  };
}

interface WidgetComponent {
  type: string;
  content: any[];
  properties: Record<string, any>;
}

export default function ProjectDetailPage() {
  const router = useRouter();
  const params = useParams();
  const project_id = params.project_id as string;

  const [userId, setUserId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [project, setProject] = useState<Project | null>(null);
  const [widgets, setWidgets] = useState<Widget[]>([]);
  const [isFetchingProject, setIsFetchingProject] = useState(false);
  const [showAddMember, setShowAddMember] = useState(false);
  const [newMemberEmail, setNewMemberEmail] = useState("");
  const [newMemberUsername, setNewMemberUsername] = useState("");
  const [inviteCode, setInviteCode] = useState("");

  useEffect(() => {
    const storedUserId = localStorage.getItem("user_id");
    if (!storedUserId) {
      router.push("/login");
    } else {
      setUserId(storedUserId);
      fetchProject(storedUserId);
      setIsLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [router, project_id]);

  const fetchProject = async (user_id: string) => {
    setIsFetchingProject(true);
    try {
      const response = await fetch(
        `http://localhost:8000/projects/view_project?project_id=${encodeURIComponent(
          project_id
        )}&user_id=${encodeURIComponent(user_id)}&project_name=&force_refresh=false`
      );
      const data = await response.json();

      if (response.ok) {
        setProject(data);
        if (data.widgets && data.widgets.length > 0) {
          await fetchWidgets(data.widgets);
        }
      } else {
        console.error("Failed to fetch project:", data);
      }
    } catch (err) {
      console.error("Error fetching project:", err);
    } finally {
      setIsFetchingProject(false);
    }
  };

  const fetchWidgets = async (widgetIds: string[]) => {
    try {
      const widgetPromises = widgetIds.map((widgetId) =>
        fetch(`http://localhost:8000/public/startup`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            user_id: userId,
            project_id: project_id,
            widget_id: widgetId,
            endpoint: "startup",
            headers: {},
            params: {},
          }),
        }).then((res) => res.json())
      );

      const widgetResults = await Promise.all(widgetPromises);
      setWidgets(widgetResults.filter((w) => w && w.widget_id));
    } catch (err) {
      console.error("Error fetching widgets:", err);
    }
  };

  const handleLikeProject = async () => {
    if (!userId || !project) return;

    try {
      const response = await fetch(`http://localhost:8000/projects/like_project`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          project_id: project_id,
          user_id: userId,
          project_name: project.project_name,
          force_refresh: false,
        }),
      });

      if (response.ok) {
        await fetchProject(userId);
      }
    } catch (err) {
      console.error("Error liking project:", err);
    }
  };

  const handleAddMember = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userId || !newMemberEmail || !newMemberUsername) return;

    try {
      const response = await fetch(
        `http://localhost:8000/projects/add_member?project_id=${encodeURIComponent(
          project_id
        )}&user_id=${encodeURIComponent(userId)}&new_email=${encodeURIComponent(
          newMemberEmail
        )}&new_username=${encodeURIComponent(newMemberUsername)}&code=${encodeURIComponent(
          inviteCode
        )}`
      );

      if (response.ok) {
        setShowAddMember(false);
        setNewMemberEmail("");
        setNewMemberUsername("");
        setInviteCode("");
        await fetchProject(userId);
      } else {
        const data = await response.json();
        alert(data.detail || "Failed to add member");
      }
    } catch (err) {
      console.error("Error adding member:", err);
    }
  };

  const handleRemoveMember = async (memberEmail: string, memberUsername: string) => {
    if (!userId || !confirm(`Remove ${memberUsername} from project?`)) return;

    try {
      const response = await fetch(
        `http://localhost:8000/projects/delete_member?project_id=${encodeURIComponent(
          project_id
        )}&user_id=${encodeURIComponent(userId)}&email=${encodeURIComponent(
          memberEmail
        )}&username=${encodeURIComponent(memberUsername)}`,
        {
          method: "DELETE",
        }
      );

      if (response.ok) {
        await fetchProject(userId);
      }
    } catch (err) {
      console.error("Error removing member:", err);
    }
  };

  const handleToggleTransparency = async () => {
    if (!userId || !project) return;

    try {
      const response = await fetch(`http://localhost:8000/projects/edit_transparency`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          project_id: project_id,
          user_id: userId,
          project_name: project.project_name,
          force_refresh: false,
          transparency: !project.project_transparency,
        }),
      });

      if (response.ok) {
        await fetchProject(userId);
      }
    } catch (err) {
      console.error("Error toggling transparency:", err);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("user_id");
    localStorage.removeItem("username");
    router.push("/");
  };

  const renderWidgetComponent = (component: WidgetComponent) => {
    switch (component.type) {
      case "button":
        return (
          <button
            key={Math.random()}
            className="px-4 py-2 bg-white hover:bg-gray-200 text-black rounded-lg transition-all"
            style={component.properties}
          >
            {component.content.join(" ")}
          </button>
        );
      case "text":
        return (
          <p key={Math.random()} className="text-white" style={component.properties}>
            {component.content.join(" ")}
          </p>
        );
      case "input":
        return (
          <input
            key={Math.random()}
            type="text"
            className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white"
            placeholder={component.content[0] || ""}
            style={component.properties}
          />
        );
      case "heading":
        return (
          <h3 key={Math.random()} className="text-xl font-bold text-white" style={component.properties}>
            {component.content.join(" ")}
          </h3>
        );
      case "list":
        return (
          <ul key={Math.random()} className="list-disc list-inside text-white" style={component.properties}>
            {component.content.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        );
      default:
        return (
          <div key={Math.random()} className="text-white/70" style={component.properties}>
            {JSON.stringify(component.content)}
          </div>
        );
    }
  };

  const getWidgetSizeClass = (size: string) => {
    switch (size) {
      case "small":
        return "col-span-1";
      case "medium":
        return "col-span-2";
      case "large":
        return "col-span-3";
      case "extra_large":
        return "col-span-4";
      default:
        return "col-span-1";
    }
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
        {isFetchingProject && !project ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin w-12 h-12 border-4 border-white/20 border-t-white rounded-full"></div>
          </div>
        ) : project ? (
          <>
            {/* Project Header */}
            <div className="mb-8">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h1 className="text-4xl font-bold text-white mb-2">
                    {project.project_name}
                  </h1>
                  <div className="flex items-center gap-4">
                    <span
                      className={`px-3 py-1 rounded-full text-sm font-medium ${
                        project.project_transparency
                          ? "bg-green-500/20 text-green-400 border border-green-500/30"
                          : "bg-red-500/20 text-red-400 border border-red-500/30"
                      }`}
                    >
                      {project.project_transparency ? "Public" : "Private"}
                    </span>
                    <button
                      onClick={handleToggleTransparency}
                      className="text-white/70 hover:text-white text-sm transition-colors"
                    >
                      Toggle Visibility
                    </button>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <button
                    onClick={handleLikeProject}
                    className="flex items-center gap-2 px-6 py-3 bg-white/10 hover:bg-white/20 border border-white/30 text-white rounded-lg transition-all"
                  >
                    <svg
                      className="w-5 h-5"
                      fill="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
                    </svg>
                    <span className="font-semibold">{project.project_likes}</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Project Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-black/50 border border-white/20 rounded-lg p-6 backdrop-blur-sm">
                <div className="flex items-center gap-3 mb-2">
                  <svg
                    className="w-6 h-6 text-white/70"
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
                  <h3 className="text-white/90 font-semibold">Members</h3>
                </div>
                <p className="text-3xl font-bold text-white">{project.project_members?.length || 0}</p>
              </div>

              <div className="bg-black/50 border border-white/20 rounded-lg p-6 backdrop-blur-sm">
                <div className="flex items-center gap-3 mb-2">
                  <svg
                    className="w-6 h-6 text-white/70"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z"
                    />
                  </svg>
                  <h3 className="text-white/90 font-semibold">Widgets</h3>
                </div>
                <p className="text-3xl font-bold text-white">{project.widgets?.length || 0}</p>
              </div>

              <div className="bg-black/50 border border-white/20 rounded-lg p-6 backdrop-blur-sm">
                <div className="flex items-center gap-3 mb-2">
                  <svg
                    className="w-6 h-6 text-white/70"
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
                  <h3 className="text-white/90 font-semibold">Organizations</h3>
                </div>
                <p className="text-3xl font-bold text-white">{project.organizations?.length || 0}</p>
              </div>
            </div>

            {/* Members Section */}
            <div className="bg-black/50 border border-white/20 rounded-lg p-6 backdrop-blur-sm mb-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">Project Members</h2>
                <button
                  onClick={() => setShowAddMember(!showAddMember)}
                  className="px-4 py-2 bg-white hover:bg-gray-200 text-black font-semibold rounded-lg transition-all flex items-center gap-2"
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

              {showAddMember && (
                <form onSubmit={handleAddMember} className="mb-6 p-4 bg-white/5 rounded-lg border border-white/10">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <input
                      type="email"
                      value={newMemberEmail}
                      onChange={(e) => setNewMemberEmail(e.target.value)}
                      placeholder="Member email"
                      className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40"
                      required
                    />
                    <input
                      type="text"
                      value={newMemberUsername}
                      onChange={(e) => setNewMemberUsername(e.target.value)}
                      placeholder="Member username"
                      className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40"
                      required
                    />
                    <input
                      type="text"
                      value={inviteCode}
                      onChange={(e) => setInviteCode(e.target.value)}
                      placeholder="Invite code (optional)"
                      className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40"
                    />
                  </div>
                  <button
                    type="submit"
                    className="w-full py-2 bg-white hover:bg-gray-200 text-black font-semibold rounded-lg transition-all"
                  >
                    Add Member
                  </button>
                </form>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {project.project_members && project.project_members.length > 0 ? (
                  project.project_members.map((member, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between p-4 bg-white/5 border border-white/10 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
                          <svg
                            className="w-6 h-6 text-white"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                            />
                          </svg>
                        </div>
                        <span className="text-white font-medium">{member}</span>
                      </div>
                      <button
                        onClick={() => handleRemoveMember("", member)}
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
                  ))
                ) : (
                  <div className="col-span-full text-center py-8 text-white/50">
                    No members yet
                  </div>
                )}
              </div>
            </div>

            {/* Widgets Section */}
            <div className="bg-black/50 border border-white/20 rounded-lg p-6 backdrop-blur-sm">
              <h2 className="text-2xl font-bold text-white mb-6">Project Widgets</h2>

              {widgets.length === 0 ? (
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
                      d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z"
                    />
                  </svg>
                  <p className="text-white/70 text-lg">No widgets yet</p>
                  <p className="text-white/50 text-sm">Create widgets to display them here</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {widgets.map((widget) => (
                    <div
                      key={widget.widget_id}
                      className={`bg-white/5 border border-white/10 rounded-lg p-6 ${getWidgetSizeClass(
                        widget.size
                      )}`}
                    >
                      <h3 className="text-xl font-bold text-white mb-4">{widget.name}</h3>
                      <div className="space-y-3">
                        {Object.entries(widget.interactions).map(([endpoint, interaction]) =>
                          interaction.components.map((component, idx) => (
                            <div key={`${endpoint}-${idx}`}>
                              {renderWidgetComponent(component)}
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="text-center py-12">
            <p className="text-white/70 text-lg">Project not found</p>
          </div>
        )}
      </div>
    </div>
  );
}
