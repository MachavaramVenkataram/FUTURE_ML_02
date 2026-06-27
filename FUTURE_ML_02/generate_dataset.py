import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_dataset(num_records=2000):
    categories = [
        "Billing Issues",
        "Technical Issues",
        "Account Access",
        "Product Inquiry",
        "General Query"
    ]

    priorities = ["High", "Medium", "Low"]

    # Sample phrases for descriptions
    phrases = {
        "Billing Issues": [
            "I was charged twice for my subscription this month.",
            "My invoice is incorrect, please fix the amount.",
            "I would like to request a refund for the recent charge.",
            "Why is there a late fee on my account?",
            "My payment method is not being accepted.",
            "I want to cancel my subscription but need to know the final bill.",
            "There's an unrecognized transaction on my credit card.",
            "How do I update my billing address?",
            "I haven't received my invoice for this billing cycle."
        ],
        "Technical Issues": [
            "The app crashes every time I try to open the dashboard.",
            "I'm getting a 500 internal server error when uploading a file.",
            "The website is very slow today and won't load images.",
            "My data is not syncing across devices.",
            "The API endpoint is returning a timeout error.",
            "I found a bug in the latest update regarding the search feature.",
            "Can you help me configure the integration with third-party tools?",
            "The software keeps freezing during the export process.",
            "I am unable to connect to the database."
        ],
        "Account Access": [
            "I cannot access my account after resetting my password.",
            "My account has been locked due to multiple login attempts.",
            "I forgot my password and the recovery email isn't arriving.",
            "How do I enable two-factor authentication on my profile?",
            "I need to change the email address associated with my account.",
            "My account was hacked and I need to regain control.",
            "Can you unlock my account? I remember my password now.",
            "The OTP for login is not being sent to my phone number.",
            "I want to delete my account permanently."
        ],
        "Product Inquiry": [
            "I would like information about your premium subscription plan.",
            "Does this software support multi-user collaboration?",
            "When will the new feature be released?",
            "Are there any discounts available for students or nonprofits?",
            "Can I get a demo of the enterprise edition?",
            "What are the limitations of the free tier?",
            "Do you offer an API for custom integrations?",
            "Is the platform compliant with GDPR?",
            "Where can I find the user manual for the latest hardware?"
        ],
        "General Query": [
            "Where are your headquarters located?",
            "Do you offer customer support in Spanish?",
            "How can I leave feedback about my experience?",
            "What are your business hours?",
            "I want to partner with your company, who should I contact?",
            "Is there a community forum for users?",
            "How long does it usually take to get a response?",
            "Can I change my notification preferences?",
            "Where do I find the terms of service?"
        ]
    }

    data = []
    start_date = datetime.now() - timedelta(days=365)

    for i in range(num_records):
        category = random.choice(categories)
        description = random.choice(phrases[category])
        
        # Add some random noise/variations to make text slightly different
        noise_words = ["Please help.", "Thanks.", "ASAP.", "Urgent!", "Any updates?", "Hello,"]
        if random.random() > 0.5:
            description = f"{random.choice(noise_words)} {description}"
        if random.random() > 0.7:
            description = f"{description} {random.choice(noise_words)}"

        # Assign priorities based roughly on categories or text context
        priority = "Low"
        if category in ["Technical Issues", "Account Access"] or "Urgent" in description or "ASAP" in description or "hacked" in description:
            priority = np.random.choice(["High", "Medium"], p=[0.7, 0.3])
        elif category in ["Billing Issues"]:
            priority = np.random.choice(["High", "Medium", "Low"], p=[0.4, 0.4, 0.2])
        else:
            priority = np.random.choice(["Medium", "Low"], p=[0.3, 0.7])

        # Generate random date
        random_days = random.randint(0, 365)
        ticket_date = start_date + timedelta(days=random_days)

        data.append({
            "Ticket ID": f"TKT-{1000 + i}",
            "Ticket Description": description,
            "Category": category,
            "Priority Level": priority,
            "Date Created": ticket_date.strftime("%Y-%m-%d %H:%M:%S")
        })

    df = pd.DataFrame(data)
    
    # Introduce some missing values and duplicates to simulate real-world data
    num_missing = int(num_records * 0.05)
    missing_indices = random.sample(range(num_records), num_missing)
    df.loc[missing_indices, "Ticket Description"] = np.nan
    
    # Append duplicates
    duplicates = df.sample(n=int(num_records * 0.03))
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv("dataset/support_tickets.csv", index=False)
    print(f"Dataset generated successfully with {len(df)} records!")

if __name__ == "__main__":
    generate_dataset()
