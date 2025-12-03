import Foundation
import Security

/// Secure storage wrapper for sensitive data like OAuth tokens
class KeychainWrapper {
    static let shared = KeychainWrapper()
    
    private init() {}
    
    // MARK: - Save
    
    func save(_ value: String, forKey key: String) -> Bool {
        guard let data = value.data(using: .utf8) else { return false }
        
        // Delete any existing item
        delete(forKey: key)
        
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlocked
        ]
        
        let status = SecItemAdd(query as CFDictionary, nil)
        return status == errSecSuccess
    }
    
    // MARK: - Retrieve
    
    func get(forKey key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        guard status == errSecSuccess,
              let data = result as? Data,
              let value = String(data: data, encoding: .utf8) else {
            return nil
        }
        
        return value
    }
    
    // MARK: - Delete
    
    func delete(forKey key: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }
    
    // MARK: - Clear All
    
    func clearAll() -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }
}

// MARK: - Keychain Keys

extension KeychainWrapper {
    static let accessTokenKey = "lazi.access_token"
    static let refreshTokenKey = "lazi.refresh_token"
    static let tokenExpiryKey = "lazi.token_expiry"
    static let userIdKey = "lazi.user_id"
    static let grantedScopesKey = "lazi.granted_scopes"
}
