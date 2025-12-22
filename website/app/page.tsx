import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <main className="relative min-h-screen w-full overflow-hidden">
      {/* Background Image */}
      <div className="absolute inset-0 z-0">
        <Image
          src="/homepage_bg.jpg"
          alt="Background"
          fill
          className="object-cover"
          priority
          quality={100}
        />
        {/* Dark overlay for better text readability */}
        <div className="absolute inset-0 bg-black/40" />
      </div>

      {/* Navigation Bar */}
      <nav className="relative z-10 w-full">
        <div className="backdrop-blur-md bg-white/10 border-b border-white/20">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              {/* Logo/Title */}
              <Link 
                href="/" 
                className="text-2xl font-bold text-white hover:text-white/80 transition-colors"
              >
                NoManager
              </Link>

              {/* Navigation Links */}
              <div className="flex items-center gap-8">
                <Link
                  href="/projects"
                  className="text-white/90 hover:text-white font-medium transition-colors"
                >
                  Projects
                </Link>
                <Link
                  href="/login"
                  className="text-white/90 hover:text-white font-medium transition-colors"
                >
                  Sign In
                </Link>
                <Link
                  href="/signup"
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-all"
                >
                  Sign Up
                </Link>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-[calc(100vh-80px)] px-6">
        <div className="text-center space-y-8">
          {/* Main Title */}
          <h1 className="text-7xl md:text-8xl lg:text-9xl font-bold text-white drop-shadow-2xl">
            NoManager
          </h1>

          {/* Subtitle */}
          <p className="text-xl md:text-2xl text-white/90 max-w-2xl mx-auto drop-shadow-lg">
            The modern way to manage your projects and widgets
          </p>

          {/* CTA Buttons */}
          <div className="pt-6 flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link
              href="/signup"
              className="inline-flex items-center justify-center px-8 py-4 text-lg font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-2xl transition-all duration-200 hover:scale-105 hover:shadow-blue-500/50"
            >
              Sign Up Free
              <svg
                className="ml-2 w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 7l5 5m0 0l-5 5m5-5H6"
                />
              </svg>
            </Link>
            <Link
              href="/login"
              className="inline-flex items-center justify-center px-8 py-4 text-lg font-semibold text-white border-2 border-white/30 hover:border-white/50 hover:bg-white/10 rounded-lg shadow-2xl transition-all duration-200 hover:scale-105"
            >
              Sign In
            </Link>
          </div>
        </div>
      </div>
    </main>
  );
}
