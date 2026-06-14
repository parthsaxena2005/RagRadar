import time
import random

TARGET_URL = http://rag_app:8000/query

QUESTION_POOL = [
    "What did NVIDIA state regarding AI inference demand and H200 GPU adoption?",
    "What are Tesla's projected capital expenditures for Gigafactory expansions and manufacturing capacity?",
    "Did Microsoft report any data center capacity constraints or AI chip supply chain issues?",
    "What were Google's stated timelines for TPU v5e or next-generation hardware infrastructure deployment?",
    "How is Amazon optimizing international fulfillment operating costs using generative AI models?",
    "What did Meta outline as their primary infrastructure investment risks and data center CAPEX projections?",
    "What specific factors drove Apple's gross margin variance in their services revenue segment?",
    "What were Netflix's subscriber acquisition costs and operating margins in the APAC region?",
    "Did Intel specify an explicit launch window or yield target for their next-generation Xeon processors?",
    "What core components drove AMD's data center segment growth and EPYC processor adoption this past quarter?"
]

while True:
    