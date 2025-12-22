"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

interface Project {
  project_id: string;
  project_name: string;
  project_transparency: boolean;
  project_likes: number;
  project_members: Array<[string, string]>;
}

interface Organization {
  id: string;
  name: string;
  members: string[];
  projects: string[];
}

export default function DashboardPage() {
  const router = useRouter();
  const [userId, setUserId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [projects, setProjects] = useState<Project[]>([]);
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [isFetchingProjects, setIsFetchingProjects] = useState(false);
  const [isFetchingOrgs, setIsFetchingOrgs] = useState(false);

  const fetchProjects = async (user_id: string) => {
    setIsFetchingProjects(true);
    try {
      const response = await fetch(
        `http://localhost:8000/projects/list?user_id=${encodeURIComponent(user_id)}&force_refresh=false`,
      );
      const data = await response.json();

      if (response.ok && Array.isArray(data)) {
        setProjects(data);
      } else {
        console.error("Failed to fetch projects:", data);
      }
    } catch (err) {
      console.error("Error fetching projects:", err);
    } finally {
      setIsFetchingProjects(false);
    }
  };

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
        },
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

  useEffect(() => {
    // Check if user is logged in
    const storedUserId = localStorage.getItem("user_id");
    if (!storedUserId) {
      // Redirect to login if not authenticated
      router.push("/login");
    } else {
      setUserId(storedUserId);
      fetchProjects(storedUserId);
      fetchOrganizations(storedUserId);
      setIsLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [router]);

  const handleLogout = () => {
    localStorage.removeItem("user_id");
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

      <div className="flex h-[calc(100vh-73px)]">
        {/* Sidebar */}
        <aside className="w-64 border-r border-white/10 bg-black/30 backdrop-blur-sm">
          <div className="p-6 space-y-2">
            {/* OAuth API Key Section */}
            <div>
              <h3 className="text-white/50 text-xs font-semibold uppercase tracking-wider mb-3">
                Developer
              </h3>
              <Link
                href="/api-keys"
                className="w-full px-4 py-3 text-left text-white/90 hover:text-white hover:bg-white/10 rounded-lg transition-all flex items-center gap-3 group"
              >
                <svg
                  className="w-5 h-5 text-white/70 group-hover:text-white transition-colors"
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
                <span className="font-medium">Get API Key</span>
              </Link>
            </div>
            {/* Organizations Section */}
            <div className="pt-6">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-white/50 text-xs font-semibold uppercase tracking-wider">
                  Organizations
                </h3>
                <Link
                  href="/organizations"
                  className="p-1 hover:bg-white/10 rounded transition-all"
                  title="View all organizations"
                >
                  <svg
                    className="w-4 h-4 text-white/50 hover:text-white"
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
                </Link>
              </div>

              {isFetchingOrgs && organizations.length === 0 ? (
                <div className="px-4 py-3 text-white/50 text-sm">
                  Loading organizations...
                </div>
              ) : organizations.length === 0 ? (
                <div className="px-4 py-3 text-white/50 text-sm">
                  No organizations yet
                </div>
              ) : (
                <div className="space-y-1 max-h-48 overflow-y-auto">
                  {organizations.map((org) => (
                    <Link
                      key={org.id}
                      href={`/organizations`}
                      className="block px-4 py-3 text-left text-white/90 hover:text-white hover:bg-white/10 rounded-lg transition-all group"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-3 min-w-0 flex-1">
                          <svg
                            className="w-5 h-5 text-white/70 group-hover:text-white transition-colors flex-shrink-0"
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
                          <span className="font-medium truncate">
                            {org.name}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-white/50 flex-shrink-0">
                          <span>{org.members.length}</span>
                          <svg
                            className="w-3 h-3"
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
                        </div>
                      </div>
                    </Link>
                  ))}
                </div>
              )}
            </div>

            {/* Projects Section */}
            {/* Projects Section */}
            <div className="pt-6">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-white/50 text-xs font-semibold uppercase tracking-wider">
                  Projects
                </h3>
                {isFetchingProjects && (
                  <div className="animate-spin w-3 h-3 border-2 border-white/20 border-t-white rounded-full"></div>
                )}
              </div>

              <Link
                href="/projects/create"
                className="w-full px-4 py-3 text-left text-white/90 hover:text-white hover:bg-white/10 rounded-lg transition-all flex items-center gap-3 group mb-3"
              >
                <svg
                  className="w-5 h-5 text-white/70 group-hover:text-white transition-colors"
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
                <span className="font-medium">Create Project</span>
              </Link>

              {/* Projects List */}
              <div className="space-y-1 max-h-[calc(100vh-400px)] overflow-y-auto">
                {projects.length === 0 && !isFetchingProjects ? (
                  <div className="px-4 py-6 text-center">
                    <svg
                      className="w-12 h-12 text-white/20 mx-auto mb-2"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"
                      />
                    </svg>
                    <p className="text-white/50 text-xs">No projects yet</p>
                  </div>
                ) : (
                  projects.map((project) => (
                    <Link
                      key={project.project_id}
                      href={`/projects/${project.project_id}`}
                      className="block px-4 py-3 text-left text-white/90 hover:text-white hover:bg-white/10 rounded-lg transition-all group"
                    >
                      <div className="flex items-center gap-3">
                        <svg
                          className="w-4 h-4 text-white/50 group-hover:text-white/70 transition-colors flex-shrink-0"
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
                        <div className="flex-1 min-w-0">
                          <p className="font-medium truncate">
                            {project.project_name}
                          </p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-xs text-white/50">
                              {project.project_transparency ? (
                                <span className="flex items-center gap-1">
                                  <svg
                                    className="w-3 h-3"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      strokeWidth={2}
                                      d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                                    />
                                  </svg>
                                  Public
                                </span>
                              ) : (
                                <span className="flex items-center gap-1">
                                  <svg
                                    className="w-3 h-3"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      strokeWidth={2}
                                      d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                                    />
                                  </svg>
                                  Private
                                </span>
                              )}
                            </span>
                          </div>
                        </div>
                      </div>
                    </Link>
                  ))
                )}
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 overflow-auto">
          <div className="p-8">
            {/* Welcome Section */}
            <div className="max-w-4xl">
              <h1 className="text-4xl font-bold text-white mb-4">
                Welcome to Your Dashboard
              </h1>
              <p className="text-white/70 text-lg mb-8">
                Get started by creating your first project or obtaining an API
                key for development.
              </p>

              {/* Quick Stats Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="bg-black/50 border border-white/20 rounded-lg p-6 backdrop-blur-sm">
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-white/10 rounded-lg">
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
                          d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"
                        />
                      </svg>
                    </div>
                    <div>
                      <p className="text-white/50 text-sm">Total Projects</p>
                      <p className="text-2xl font-bold text-white">0</p>
                    </div>
                  </div>
                </div>

                <div className="bg-black/50 border border-white/20 rounded-lg p-6 backdrop-blur-sm">
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-white/10 rounded-lg">
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
                          d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01"
                        />
                      </svg>
                    </div>
                    <div>
                      <p className="text-white/50 text-sm">Active Widgets</p>
                      <p className="text-2xl font-bold text-white">0</p>
                    </div>
                  </div>
                </div>

                <div className="bg-black/50 border border-white/20 rounded-lg p-6 backdrop-blur-sm">
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-white/10 rounded-lg">
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
                          d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"
                        />
                      </svg>
                    </div>
                    <div>
                      <p className="text-white/50 text-sm">API Keys</p>
                      <p className="text-2xl font-bold text-white">0</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Getting Started Section */}
              <div className="bg-black/50 border border-white/20 rounded-lg p-8 backdrop-blur-sm">
                <h2 className="text-2xl font-bold text-white mb-4">
                  Getting Started
                </h2>
                <div className="space-y-4">
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0 w-8 h-8 bg-white text-black rounded-full flex items-center justify-center font-bold">
                      1
                    </div>
                    <div>
                      <h3 className="text-white font-semibold mb-1">
                        Get Your API Key
                      </h3>
                      <p className="text-white/70">
                        Obtain your developer API key to start building widgets
                        and integrations.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0 w-8 h-8 bg-white text-black rounded-full flex items-center justify-center font-bold">
                      2
                    </div>
                    <div>
                      <h3 className="text-white font-semibold mb-1">
                        Create Your First Project
                      </h3>
                      <p className="text-white/70">
                        Set up a new project to organize your widgets and manage
                        your workspace.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0 w-8 h-8 bg-white text-black rounded-full flex items-center justify-center font-bold">
                      3
                    </div>
                    <div>
                      <h3 className="text-white font-semibold mb-1">
                        Build Custom Widgets
                      </h3>
                      <p className="text-white/70">
                        Use our SDK to create interactive widgets tailored to
                        your needs.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
