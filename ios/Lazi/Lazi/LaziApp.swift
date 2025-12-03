//
//  LaziApp.swift
//  Lazi
//
//  Created by Jacey Simpson on 9/1/25.
//

import SwiftUI

@main
struct LaziApp: App {
    @StateObject private var oauthService = OAuthService.shared
    
    var body: some Scene {
        WindowGroup {
            if oauthService.isAuthenticated {
                // User is authenticated - show main app
                ContentView()
                    .environmentObject(oauthService)
            } else {
                // User not authenticated - show OAuth login
                OAuthLoginView()
                    .environmentObject(oauthService)
            }
        }
    }
}
