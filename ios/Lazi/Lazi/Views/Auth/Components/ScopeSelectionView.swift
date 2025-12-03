import SwiftUI

/// Component for selecting OAuth2 scopes during login
struct ScopeSelectionView: View {
    @Binding var selectedScopes: Set<OAuthScope>
    @State private var isExpanded = false
    
    private var scopesByCategory: [String: [OAuthScope]] {
        Dictionary(grouping: OAuthScope.allCases, by: { $0.category })
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            Button(action: { isExpanded.toggle() }) {
                HStack {
                    Text("Permissions")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Spacer()
                    
                    Text("\(selectedScopes.count) selected")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .foregroundColor(.secondary)
                        .font(.caption)
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(10)
            
            // Expanded scope list
            if isExpanded {
                VStack(alignment: .leading, spacing: 16) {
                    ForEach(scopesByCategory.keys.sorted(), id: \.self) { category in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(category)
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(.secondary)
                            
                            ForEach(scopesByCategory[category] ?? [], id: \.id) { scope in
                                ScopeRow(
                                    scope: scope,
                                    isSelected: selectedScopes.contains(scope),
                                    onToggle: {
                                        if selectedScopes.contains(scope) {
                                            selectedScopes.remove(scope)
                                        } else {
                                            selectedScopes.insert(scope)
                                        }
                                    }
                                )
                            }
                        }
                    }
                }
                .padding()
                .background(Color(.systemBackground))
                .cornerRadius(10)
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color(.systemGray4), lineWidth: 1)
                )
            }
        }
    }
}

// MARK: - Scope Row

private struct ScopeRow: View {
    let scope: OAuthScope
    let isSelected: Bool
    let onToggle: () -> Void
    
    var body: some View {
        Button(action: onToggle) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: isSelected ? "checkmark.square.fill" : "square")
                    .foregroundColor(isSelected ? .blue : .secondary)
                    .font(.title3)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(scope.displayName)
                        .font(.body)
                        .foregroundColor(.primary)
                    
                    Text(scope.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                
                Spacer()
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

#Preview {
    ScopeSelectionView(selectedScopes: .constant([
        .widgetsRead,
        .projectsRead
    ]))
    .padding()
}
