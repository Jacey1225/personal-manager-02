import SwiftUI

/// Main OAuth2 login and registration view
struct OAuthLoginView: View {
    @StateObject private var viewModel = AuthViewModel()
    @EnvironmentObject var oauthService: OAuthService
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Logo/Header
                    headerSection
                    
                    // Form
                    formSection
                    
                    // Scope Selection (only visible when expanded)
                    ScopeSelectionView(selectedScopes: $viewModel.selectedScopes)
                    
                    // Submit Button
                    submitButton
                    
                    // Mode Toggle
                    modeToggleButton
                    
                    Spacer()
                }
                .padding()
            }
            .navigationTitle(viewModel.isLoginMode ? "Sign In" : "Create Account")
            .navigationBarTitleDisplayMode(.large)
            .alert("Error", isPresented: $viewModel.showError) {
                Button("OK", role: .cancel) {
                    viewModel.errorMessage = nil
                }
            } message: {
                if let errorMessage = viewModel.errorMessage {
                    Text(errorMessage)
                }
            }
        }
    }
    
    // MARK: - Header Section
    
    private var headerSection: some View {
        VStack(spacing: 12) {
            Image(systemName: "lock.shield.fill")
                .font(.system(size: 60))
                .foregroundColor(.blue)
            
            Text(viewModel.isLoginMode ? "Welcome Back" : "Join Lazi")
                .font(.title)
                .fontWeight(.bold)
            
            Text(viewModel.isLoginMode ?
                 "Sign in to access your projects and widgets" :
                 "Create an account to get started")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(.vertical)
    }
    
    // MARK: - Form Section
    
    private var formSection: some View {
        VStack(spacing: 16) {
            // Username
            AuthTextField(
                title: "Username",
                placeholder: "Enter your username",
                text: $viewModel.username,
                icon: "person.fill"
            )
            
            // Email (signup only)
            if !viewModel.isLoginMode {
                AuthTextField(
                    title: "Email",
                    placeholder: "your.email@example.com",
                    text: $viewModel.email,
                    keyboardType: .emailAddress,
                    icon: "envelope.fill"
                )
            }
            
            // Password
            AuthTextField(
                title: "Password",
                placeholder: "Enter your password",
                text: $viewModel.password,
                isSecure: true,
                icon: "lock.fill"
            )
            
            // Confirm Password (signup only)
            if !viewModel.isLoginMode {
                AuthTextField(
                    title: "Confirm Password",
                    placeholder: "Re-enter your password",
                    text: $viewModel.confirmPassword,
                    isSecure: true,
                    icon: "lock.fill",
                    errorMessage: viewModel.passwordsMatch ? nil : "Passwords do not match"
                )
            }
        }
    }
    
    // MARK: - Submit Button
    
    private var submitButton: some View {
        Button(action: {
            Task {
                if viewModel.isLoginMode {
                    await viewModel.login()
                } else {
                    await viewModel.signup()
                }
            }
        }) {
            HStack {
                if viewModel.isLoading {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                } else {
                    Text(viewModel.isLoginMode ? "Sign In" : "Create Account")
                        .fontWeight(.semibold)
                }
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(viewModel.isFormValid ? Color.blue : Color.gray)
            .foregroundColor(.white)
            .cornerRadius(10)
        }
        .disabled(!viewModel.isFormValid || viewModel.isLoading)
    }
    
    // MARK: - Mode Toggle Button
    
    private var modeToggleButton: some View {
        Button(action: { viewModel.toggleMode() }) {
            HStack(spacing: 4) {
                Text(viewModel.isLoginMode ? "Don't have an account?" : "Already have an account?")
                    .foregroundColor(.secondary)
                Text(viewModel.isLoginMode ? "Sign Up" : "Sign In")
                    .foregroundColor(.blue)
                    .fontWeight(.semibold)
            }
            .font(.subheadline)
        }
        .padding(.top, 8)
    }
}

#Preview {
    OAuthLoginView()
        .environmentObject(OAuthService.shared)
}
