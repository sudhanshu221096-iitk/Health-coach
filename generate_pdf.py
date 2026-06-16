"""Generate wellness_protocol.pdf in data/ folder."""
from fpdf import FPDF

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Helvetica", "B", 20)
pdf.cell(0, 12, "Wellness Protocol Guide", ln=True, align="C")
pdf.ln(5)

sections = [
    ("1. Sleep Optimization", [
        "Sleep 7-9 hours per night. Maintain a consistent sleep schedule even on weekends.",
        "Avoid screens (phones, laptops, TV) at least 60 minutes before bedtime.",
        "Keep bedroom temperature between 65-68 degrees Fahrenheit (18-20 Celsius).",
        "Avoid caffeine after 2 PM. Limit alcohol as it disrupts deep sleep cycles.",
        "Use blackout curtains or a sleep mask to block light. Use earplugs if needed.",
        "Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s. Repeat 4 times before bed.",
        "Expose yourself to natural sunlight within 30 minutes of waking to set circadian rhythm.",
    ]),
    ("2. Nutrition Guidelines", [
        "Eat whole foods: vegetables, fruits, lean proteins, whole grains, healthy fats.",
        "Aim for 5 servings of vegetables and 2 servings of fruit daily.",
        "Drink at least 8 glasses (2 litres) of water per day. More if exercising.",
        "Eat breakfast within 1 hour of waking. Do not skip breakfast.",
        "Avoid processed foods, added sugars, and refined carbohydrates.",
        "Eat slowly. Chew each bite 20-30 times. Stop eating when 80% full.",
        "Do not eat within 2-3 hours of bedtime to support digestion and sleep.",
        "Limit sodium intake to under 2300mg per day.",
        "Include omega-3 rich foods: salmon, walnuts, flaxseeds, chia seeds.",
    ]),
    ("3. Exercise Protocol", [
        "Aim for at least 150 minutes of moderate exercise per week (30 min x 5 days).",
        "Include both cardio (walking, running, cycling) and strength training (2x per week).",
        "Walk at least 8000-10000 steps per day.",
        "Warm up for 5-10 minutes before every workout. Cool down and stretch afterward.",
        "Day 1-2: Light walking, 20-30 minutes. Focus on form and consistency.",
        "Day 3-5: Add bodyweight exercises: squats, push-ups, lunges (3 sets of 10).",
        "Day 6-7: Rest or light yoga/stretching. Allow muscles to recover.",
        "Week 2 onwards: Gradually increase intensity by 10% each week.",
        "Never exercise to exhaustion in the first two weeks. Build the habit first.",
    ]),
    ("4. Stress Management", [
        "Practice mindfulness meditation for 10 minutes daily — morning is best.",
        "Keep a gratitude journal. Write 3 things you are grateful for each day.",
        "Limit news and social media consumption to 30 minutes per day.",
        "Take short breaks (5 minutes) every 90 minutes during work.",
        "Connect with friends or family for at least 15 minutes daily.",
        "Deep breathing: 5 slow breaths before any stressful situation.",
        "Avoid multitasking. Focus on one task at a time to reduce mental load.",
    ]),
    ("5. Hydration Rules", [
        "Drink a full glass of water (250ml) immediately upon waking.",
        "Drink water before every meal — this aids digestion and reduces overeating.",
        "Carry a water bottle at all times. Set reminders to drink every hour.",
        "Herbal teas count toward daily hydration. Caffeinated drinks do not.",
        "Urine should be pale yellow. Dark yellow means you are dehydrated.",
        "Increase water intake by 500ml for every 30 minutes of exercise.",
    ]),
    ("6. Tracking & Check-In Rules", [
        "Check in with your health coach every day for the first 30 days.",
        "Track sleep hours, water intake, exercise, and mood daily.",
        "Rate your energy levels daily from 1-10. Note patterns over time.",
        "Report any unusual symptoms, fatigue, or pain immediately.",
        "Weekly weigh-in: weigh yourself on the same day, same time, same conditions.",
        "Take monthly progress photos in the same pose and lighting for comparison.",
        "Review goals every Sunday. Adjust habits that are not working.",
    ]),
    ("7. Dos and Don'ts", [
        "DO: Start small. One habit at a time. Consistency beats intensity.",
        "DO: Celebrate small wins. Progress is not always linear.",
        "DO: Sleep and recovery are as important as exercise and nutrition.",
        "DO: Communicate openly with your coach about struggles.",
        "DON'T: Skip meals to lose weight. This backfires and slows metabolism.",
        "DON'T: Compare your progress to others. Everyone's body is different.",
        "DON'T: Quit after one bad day. Get back on track the next meal or workout.",
        "DON'T: Take supplements without consulting a healthcare provider first.",
        "DON'T: Exercise when sick or severely fatigued. Rest is productive.",
    ]),
]

for title, points in sections:
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("Helvetica", "", 11)
    for point in points:
        pdf.multi_cell(0, 7, f"  • {point}")
        pdf.ln(1)
    pdf.ln(4)

pdf.output("/home/claude/healthcoach/data/wellness_protocol.pdf")
print("PDF created successfully.")
