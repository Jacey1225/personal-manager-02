import Foundation

print("Enter your name:")
if let name = readLine() {
    for i in 1...3 {
        print("Hello, \(name)! This is message number \(i).")
    }
}