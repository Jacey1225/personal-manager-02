import SwiftUI

/// Reusable styled text field for authentication forms
struct AuthTextField: View {
    let title: String
    let placeholder: String
    @Binding var text: String
    var isSecure: Bool = false
    var keyboardType: UIKeyboardType = .default
    var autocapitalization: TextInputAutocapitalization = .never
    var icon: String? = nil
    var errorMessage: String? = nil
    
    @State private var isPasswordVisible = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Label
            Text(title)
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(.primary)
            
            // Text Field Container
            HStack(spacing: 12) {
                // Icon
                if let icon = icon {
                    Image(systemName: icon)
                        .foregroundColor(.secondary)
                        .frame(width: 20)
                }
                
                // Text Field
                if isSecure && !isPasswordVisible {
                    SecureField(placeholder, text: $text)
                        .textInputAutocapitalization(autocapitalization)
                        .keyboardType(keyboardType)
                } else {
                    TextField(placeholder, text: $text)
                        .textInputAutocapitalization(autocapitalization)
                        .keyboardType(keyboardType)
                        .autocorrectionDisabled()
                }
                
                // Password visibility toggle
                if isSecure {
                    Button(action: { isPasswordVisible.toggle() }) {
                        Image(systemName: isPasswordVisible ? "eye.slash.fill" : "eye.fill")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(10)
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(errorMessage != nil ? Color.red : Color.clear, lineWidth: 1)
            )
            
            // Error Message
            if let errorMessage = errorMessage {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.circle.fill")
                        .font(.caption)
                    Text(errorMessage)
                        .font(.caption)
                }
                .foregroundColor(.red)
            }
        }
    }
}

#Preview {
    VStack(spacing: 20) {
        AuthTextField(
            title: "Username",
            placeholder: "Enter your username",
            text: .constant(""),
            icon: "person.fill"
        )
        
        AuthTextField(
            title: "Email",
            placeholder: "your.email@example.com",
            text: .constant(""),
            keyboardType: .emailAddress,
            icon: "envelope.fill"
        )
        
        AuthTextField(
            title: "Password",
            placeholder: "Enter your password",
            text: .constant(""),
            isSecure: true,
            icon: "lock.fill",
            errorMessage: "Password must be at least 8 characters"
        )
    }
    .padding()
}
