import SwiftUI

// MARK: - Calendar Data Models

struct CalendarMonth: Identifiable {
    let id = UUID()
    let month: Int
    let year: Int
    let name: String
    let days: [CalendarDay]
}

struct CalendarDay: Identifiable {
    let id = UUID()
    let day: Int
    let month: Int
    let year: Int
    let hasEvents: Bool
}

// MARK: - Task Calendar View Widget

struct TaskCalendarView: View {
    let userId: String
    let onDateSelected: (String) -> Void
    
    @State private var selectedMonth: Int = Calendar.current.component(.month, from: Date())
    @State private var selectedYear: Int = Calendar.current.component(.year, from: Date())
    @State private var showingMonthPicker = false
    @State private var isExpanded = false
    
    private let months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    
    private let calendar = Calendar.current
    
    var body: some View {
        VStack(spacing: 0) {
            // Header with month/year selector
            calendarHeader
            
            if isExpanded {
                calendarGrid
            }
        }
        .background(calendarBackground)
        .overlay(calendarBorder)
        .sheet(isPresented: $showingMonthPicker) {
            MonthYearPickerSheet(
                selectedMonth: $selectedMonth,
                selectedYear: $selectedYear,
                onMonthSelected: { month, year in
                    selectedMonth = month
                    selectedYear = year
                    selectMonth(month, year: year)
                }
            )
        }
    }
    
    // MARK: - View Components
    
    private var calendarHeader: some View {
        HStack {
            monthYearButton
            Spacer()
            expandButton
        }
        .padding()
    }
    
    private var monthYearButton: some View {
        Button(action: {
            showingMonthPicker = true
        }) {
            HStack {
                Text("\(months[selectedMonth - 1]) \(selectedYear)")
                    .font(.headline)
                    .fontWeight(.semibold)
                
                Image(systemName: "chevron.down")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .foregroundColor(.primary)
    }
    
    private var expandButton: some View {
        Button(action: {
            withAnimation(.easeInOut(duration: 0.3)) {
                isExpanded.toggle()
            }
        }) {
            Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                .font(.title3)
                .foregroundColor(.secondary)
        }
    }
    
    private var calendarGrid: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 7), spacing: 8) {
            weekDayHeaders
            calendarDays
        }
        .padding(.horizontal)
        .padding(.bottom)
        .transition(.opacity.combined(with: .slide))
    }
    
    private var weekDayHeaders: some View {
        ForEach(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], id: \.self) { day in
            Text(day)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.secondary)
                .frame(height: 30)
        }
    }
    
    private var calendarDays: some View {
        ForEach(getDaysInMonth(), id: \.id) { day in
            if day.day == 0 {
                // Empty cell for padding
                Rectangle()
                    .fill(Color.clear)
                    .frame(height: 35)
            } else {
                CalendarDayButton(
                    day: day,
                    isToday: isToday(day.day),
                    onTap: { selectDay(day.day) }
                )
            }
        }
    }
    
    private var calendarBackground: some View {
        RoundedRectangle(cornerRadius: 12)
            .fill(Color.black.opacity(0.05))
    }
    
    private var calendarBorder: some View {
        RoundedRectangle(cornerRadius: 12)
            .stroke(Color.white.opacity(0.8), lineWidth: 1)
    }
    
    // MARK: - Calendar Helper Functions
    
    private func getDaysInMonth() -> [CalendarDay] {
        var days: [CalendarDay] = []
        
        // Get the first day of the month
        let firstOfMonth = calendar.date(from: DateComponents(year: selectedYear, month: selectedMonth, day: 1))!
        
        // Get the weekday of the first day (0 = Sunday, 1 = Monday, etc.)
        let firstWeekday = calendar.component(.weekday, from: firstOfMonth) - 1
        
        // Add empty cells for padding
        for _ in 0..<firstWeekday {
            days.append(CalendarDay(day: 0, month: selectedMonth, year: selectedYear, hasEvents: false))
        }
        
        // Get the number of days in the month
        let daysInMonth = calendar.range(of: .day, in: .month, for: firstOfMonth)!.count
        
        // Add the actual days
        for day in 1...daysInMonth {
            // TODO: Implement event checking logic
            let hasEvents = false // This will be implemented when we integrate with event data
            days.append(CalendarDay(day: day, month: selectedMonth, year: selectedYear, hasEvents: hasEvents))
        }
        
        return days
    }
    
    private func isToday(_ day: Int) -> Bool {
        let today = Date()
        let todayDay = calendar.component(.day, from: today)
        let todayMonth = calendar.component(.month, from: today)
        let todayYear = calendar.component(.year, from: today)
        
        return day == todayDay && selectedMonth == todayMonth && selectedYear == todayYear
    }
    
    private func selectDay(_ day: Int) {
        if let date = calendar.date(from: DateComponents(year: selectedYear, month: selectedMonth, day: day)) {
            let targetStart = formatDateToISO(date)
            print("Selected day: \(targetStart)")
            onDateSelected(targetStart)
        }
    }
    
    private func selectMonth(_ month: Int, year: Int) {
        if let date = calendar.date(from: DateComponents(year: year, month: month, day: 1)) {
            let targetStart = formatDateToISO(date)
            print("Selected month: \(targetStart)")
            onDateSelected(targetStart)
        }
    }
    
    // MARK: - Helper Functions
    
    private func formatDateToISO(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }
}

// MARK: - Calendar Day Button Component

struct CalendarDayButton: View {
    let day: CalendarDay
    let isToday: Bool
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            dayContent
        }
        .buttonStyle(PlainButtonStyle())
    }
    
    private var dayContent: some View {
        VStack(spacing: 2) {
            Text("\(day.day)")
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(textColor)
            
            if day.hasEvents {
                eventIndicator
            }
        }
        .frame(width: 35, height: 35)
        .background(backgroundCircle)
        .overlay(borderCircle)
    }
    
    private var textColor: Color {
        isToday ? .white : .primary
    }
    
    private var eventIndicator: some View {
        Circle()
            .fill(Color.blue)
            .frame(width: 4, height: 4)
    }
    
    private var backgroundCircle: some View {
        Circle()
            .fill(isToday ? Color.blue : Color.clear)
    }
    
    private var borderCircle: some View {
        Circle()
            .stroke(Color.blue.opacity(0.3), lineWidth: 1)
            .opacity(day.hasEvents ? 1 : 0)
    }
}

// MARK: - Month/Year Picker Sheet

struct MonthYearPickerSheet: View {
    @Binding var selectedMonth: Int
    @Binding var selectedYear: Int
    let onMonthSelected: (Int, Int) -> Void
    
    @Environment(\.presentationMode) var presentationMode
    @State private var tempMonth: Int
    @State private var tempYear: Int
    
    private let months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    
    private let years = Array(2020...2030) // Adjust range as needed
    
    init(selectedMonth: Binding<Int>, selectedYear: Binding<Int>, onMonthSelected: @escaping (Int, Int) -> Void) {
        self._selectedMonth = selectedMonth
        self._selectedYear = selectedYear
        self.onMonthSelected = onMonthSelected
        self._tempMonth = State(initialValue: selectedMonth.wrappedValue)
        self._tempYear = State(initialValue: selectedYear.wrappedValue)
    }
    
    var body: some View {
        NavigationView {
            VStack {
                HStack {
                    // Month picker
                    VStack {
                        Text("Month")
                            .font(.headline)
                            .padding(.bottom, 8)
                        
                        Picker("Month", selection: $tempMonth) {
                            ForEach(1...12, id: \.self) { month in
                                Text(months[month - 1])
                                    .tag(month)
                            }
                        }
                        .pickerStyle(WheelPickerStyle())
                        .frame(width: 150)
                    }
                    
                    // Year picker
                    VStack {
                        Text("Year")
                            .font(.headline)
                            .padding(.bottom, 8)
                        
                        Picker("Year", selection: $tempYear) {
                            ForEach(years, id: \.self) { year in
                                Text("\(year)")
                                    .tag(year)
                            }
                        }
                        .pickerStyle(WheelPickerStyle())
                        .frame(width: 100)
                    }
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("Select Date")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                leading: Button("Cancel") {
                    presentationMode.wrappedValue.dismiss()
                },
                trailing: Button("Done") {
                    selectedMonth = tempMonth
                    selectedYear = tempYear
                    onMonthSelected(tempMonth, tempYear)
                    presentationMode.wrappedValue.dismiss()
                }
            )
        }
    }
}

// MARK: - Preview

struct TaskCalendarView_Previews: PreviewProvider {
    static var previews: some View {
        TaskCalendarView(userId: "preview-user") { targetStart in
            print("Selected date: \(targetStart)")
        }
        .padding()
        .previewLayout(.sizeThatFits)
    }
}
