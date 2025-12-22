import Link from "next/link";

export default function ProjectsPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800">
      {/* Navigation Bar */}
      <nav className="w-full border-b border-white/10">
        <div className="backdrop-blur-md bg-white/5">
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
                  className="text-white font-medium"
                >
                  Projects
                </Link>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-16">
        <div className="text-center space-y-6">
          <h1 className="text-5xl font-bold text-white">
            Projects
          </h1>
          <p className="text-xl text-gray-300">
            Coming soon... This is where you'll manage all your projects.
          </p>
          
          <div className="pt-8">
            <Link
              href="/"
              className="inline-flex items-center justify-center px-6 py-3 text-white/80 hover:text-white border border-white/20 hover:border-white/40 rounded-lg transition-all"
            >
              <svg
                className="mr-2 w-5 h-5"
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
              Back to Home
            </Link>
          </div>
        </div>
      </div>
    </main>
  );
}
