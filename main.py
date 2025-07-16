# simplified_customer_chatbot.py

import os
import re
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Configure Gemini API
api_key = st.secrets["gemini_key"]
genai.configure(api_key=api_key)

class CustomerQueryType(Enum):
    """Types of customer queries"""
    PAYMENT_INQUIRY = "payment_inquiry"
    CONTRACT_QUESTION = "contract_question"
    MAINTENANCE_REQUEST = "maintenance_request"
    GENERAL_POLICY = "general_policy"
    LEASE_INFO = "lease_info"
    CONTACT_INFO = "contact_info"
    AMENITIES = "amenities"
    PROCEDURES = "procedures"

class CustomerDataRetriever:
    """Retrieve general data for customer queries"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_general_data(self, query_type: CustomerQueryType, query: str) -> pd.DataFrame:
        """Get relevant data based on query type"""
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            if query_type == CustomerQueryType.PAYMENT_INQUIRY:
                # Get sample payment information (anonymized)
                df = pd.read_sql_query("""
                    SELECT payment_type, AVG(amount) as avg_amount, 
                           method, COUNT(*) as frequency
                    FROM payments 
                    GROUP BY payment_type, method
                    ORDER BY frequency DESC
                """, conn)
                
            elif query_type == CustomerQueryType.LEASE_INFO:
                # Get general lease information
                df = pd.read_sql_query("""
                    SELECT AVG(rent_amount) as avg_rent, 
                           AVG(security_deposit) as avg_deposit,
                           COUNT(*) as total_units
                    FROM leases 
                    WHERE status = 'active'
                """, conn)
                
            elif query_type == CustomerQueryType.MAINTENANCE_REQUEST:
                # Get maintenance statistics
                df = pd.read_sql_query("""
                    SELECT category, subcategory, priority, 
                           COUNT(*) as frequency,
                           AVG(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completion_rate
                    FROM service_tickets 
                    GROUP BY category, subcategory, priority
                    ORDER BY frequency DESC
                """, conn)
                
            else:
                # Return empty dataframe for other types
                df = pd.DataFrame()
                
        except Exception as e:
            st.error(f"Database error: {e}")
            df = pd.DataFrame()
        finally:
            conn.close()
            
        return df

class CustomerKnowledgeBase:
    """Customer-focused knowledge base"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_docs = []
        self.vector_store = None
        self._initialize_customer_knowledge()
    
    def _initialize_customer_knowledge(self):
        """Initialize with comprehensive customer knowledge"""
        
        customer_knowledge = [
            """
            RENT PAYMENT INFORMATION:
            - Rent is due on the 1st of each month
            - Late fees of $50 apply after the 5th of the month
            - Payment methods accepted: Online portal, bank transfer, check, money order, credit/debit card
            - Online payments can be made 24/7 through the tenant portal
            - Automatic payment setup is available to avoid late fees
            - Payment confirmation receipts are sent via email
            - For payment issues or questions, contact the office during business hours
            - Security deposits are held in escrow and returned within 30 days of move-out (minus any deductions)
            """,
            
            """
            LEASE CONTRACT AND TERMS:
            - Standard lease terms are 12 months unless otherwise specified
            - Month-to-month leases available with management approval
            - Rent increases require 60 days written notice as per state law
            - Lease renewals are typically offered 90 days before expiration
            - Early termination requires 60 days notice and may include termination fee
            - Subletting requires written permission from management
            - Lease violations may result in notices or termination
            - All lease modifications must be in writing and signed by both parties
            - Tenant has right to peaceful enjoyment of the property
            """,
            
            """
            MAINTENANCE AND REPAIRS:
            - Emergency maintenance (water leaks, no heat/AC, electrical hazards): Call emergency line 24/7
            - Urgent maintenance (non-working appliances, plumbing issues): Submit online, 24-48 hour response
            - Standard maintenance (general repairs, cosmetic issues): Submit online, 3-5 business day response
            - Maintenance request categories: Plumbing, Electrical, HVAC, Appliances, General Repair, Pest Control
            - Routine maintenance scheduled with 24-48 hours advance notice
            - Tenants must provide reasonable access for repairs
            - Emergency contact number: (555) 123-EMERGENCY
            - Maintenance hours: Monday-Friday 8AM-5PM, emergency services available 24/7
            """,
            
            """
            TENANT RIGHTS AND RESPONSIBILITIES:
            - Right to habitable living conditions and timely repairs
            - Right to privacy with proper notice before entry (except emergencies)
            - Right to quiet enjoyment without unreasonable interference
            - Responsibility to maintain cleanliness and report damages promptly
            - Responsibility to follow all lease terms and property rules
            - Responsibility to not disturb other tenants
            - Right to request maintenance and repairs in writing
            - Responsibility to allow access for necessary maintenance and inspections
            """,
            
            """
            PROPERTY POLICIES AND RULES:
            - Quiet hours: 10PM - 8AM on weekdays, 11PM - 9AM on weekends
            - Parking: One assigned space per unit, guest parking available (2-hour limit)
            - Pets: Pet policy varies by property, additional deposit and monthly fee required
            - Smoking: Prohibited in all indoor areas and within 25 feet of buildings
            - Guest policy: Guests staying more than 14 days require management approval
            - Pool and amenity hours: 6AM - 10PM daily
            - Gym access requires key fob, hours posted at facility
            - Common area guidelines posted in each facility
            """,
            
            """
            CONTACT INFORMATION AND OFFICE PROCEDURES:
            - Office hours: Monday-Friday 8AM-6PM, Saturday 9AM-4PM, Closed Sundays
            - Main office phone: (555) 123-4567
            - Emergency maintenance: (555) 123-EMERGENCY
            - Email: info@propertymanagement.com
            - Online portal available 24/7 for payments and maintenance requests
            - Address changes must be reported within 30 days
            - Key replacements: $25 fee, available during office hours
            - Move-in/move-out inspections required, schedule with office
            - After-hours package pickup available with prior arrangement
            """,
            
            """
            AMENITIES AND SERVICES:
            - Swimming pool with seasonal hours (May-September)
            - Fitness center with cardio and weight equipment
            - Community room available for resident events
            - On-site laundry facilities (coin-operated)
            - Package receiving service during office hours
            - Professional landscaping and grounds maintenance
            - 24/7 emergency maintenance response
            - Online resident portal for payments and requests
            - Referral bonus program for new tenants
            """,
            
            """
            MOVE-IN AND MOVE-OUT PROCEDURES:
            - Move-in inspection must be completed within 48 hours
            - Security deposit receipt provided at lease signing
            - Utility connection information provided at move-in
            - Move-out notice required 30-60 days in advance (check lease)
            - Move-out inspection scheduled within 72 hours of vacancy
            - Security deposit return within 30 days with itemized deductions
            - Forwarding address required for deposit return
            - Professional cleaning recommended but not required
            - Key return required to avoid lock change fees
            """
        ]
        
        # Create document embeddings
        for i, doc in enumerate(customer_knowledge):
            self.knowledge_docs.append({
                'content': doc,
                'id': f'knowledge_{i}',
                'type': 'policy'
            })
        
        # Build vector store
        if self.knowledge_docs:
            texts = [doc['content'] for doc in self.knowledge_docs]
            embeddings = self.embedding_model.encode(texts)
            
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(embeddings.astype('float32'))
    
    def search_knowledge(self, query: str, k: int = 3) -> List[str]:
        """Search knowledge base for relevant information"""
        if not self.vector_store or not self.knowledge_docs:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.vector_store.search(query_embedding.astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.knowledge_docs):
                results.append(self.knowledge_docs[idx]['content'])
        
        return results

class SimplifiedCustomerChatbot:
    """Simplified customer chatbot without authentication"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.data_retriever = CustomerDataRetriever(db_path)
        self.knowledge_base = CustomerKnowledgeBase()
        
        self.model = genai.GenerativeModel(
            "gemini-2.0-flash-exp",
            system_instruction=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        return """
        You are a helpful customer service assistant for a property management company. 
        You provide information about:
        
        1. GENERAL POLICIES: Rent payment, lease terms, property rules
        2. MAINTENANCE: How to submit requests, emergency procedures, response times
        3. AMENITIES: Available facilities, hours, access procedures
        4. PROCEDURES: Move-in/out, contact information, office hours
        
        GUIDELINES:
        - Be friendly, professional, and helpful
        - Provide accurate information based on knowledge base
        - Give general guidance without accessing specific tenant data
        - Direct customers to contact the office for account-specific questions
        - Use simple, clear language
        - Be empathetic and understanding
        
        LIMITATIONS:
        - Cannot access individual tenant accounts or specific data
        - Cannot make payment arrangements or modifications
        - Cannot schedule maintenance or guarantee response times
        - Cannot modify lease terms or policies
        
        Focus on providing helpful general information that applies to all tenants.
        """
    
    def classify_query(self, query: str) -> CustomerQueryType:
        """Classify the type of customer query"""
        query_lower = query.lower()
        
        # Payment-related keywords
        payment_keywords = ['payment', 'pay', 'rent', 'due', 'late fee', 'deposit', 'money', 'bill', 'cost']
        if any(keyword in query_lower for keyword in payment_keywords):
            return CustomerQueryType.PAYMENT_INQUIRY
        
        # Contract/lease keywords
        contract_keywords = ['lease', 'contract', 'renewal', 'term', 'policy', 'rule', 'agreement', 'breaking lease', 'subletting']
        if any(keyword in query_lower for keyword in contract_keywords):
            return CustomerQueryType.CONTRACT_QUESTION
        
        # Maintenance keywords
        maintenance_keywords = ['maintenance', 'repair', 'fix', 'broken', 'not working', 'issue', 'problem', 'emergency']
        if any(keyword in query_lower for keyword in maintenance_keywords):
            return CustomerQueryType.MAINTENANCE_REQUEST
        
        # Contact/office keywords
        contact_keywords = ['contact', 'office', 'phone', 'email', 'hours', 'address', 'call']
        if any(keyword in query_lower for keyword in contact_keywords):
            return CustomerQueryType.CONTACT_INFO
        
        # Amenities keywords
        amenity_keywords = ['pool', 'gym', 'fitness', 'laundry', 'parking', 'amenities', 'facilities']
        if any(keyword in query_lower for keyword in amenity_keywords):
            return CustomerQueryType.AMENITIES
        
        # Procedures keywords
        procedure_keywords = ['move in', 'move out', 'inspection', 'keys', 'utilities', 'process', 'procedure']
        if any(keyword in query_lower for keyword in procedure_keywords):
            return CustomerQueryType.PROCEDURES
        
        return CustomerQueryType.GENERAL_POLICY
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process customer query and return response"""
        
        query_type = self.classify_query(query)
        
        # Search knowledge base for relevant information
        relevant_knowledge = self.knowledge_base.search_knowledge(query)
        
        # Get general data if applicable
        general_data = self.data_retriever.get_general_data(query_type, query)
        
        # Generate response
        return self._generate_response(query, query_type, relevant_knowledge, general_data)
    
    def _generate_response(self, query: str, query_type: CustomerQueryType, 
                         knowledge: List[str], data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive response"""
        
        # Prepare data context
        data_context = ""
        if not data.empty:
            data_context = f"Additional data context: {data.to_dict('records')}"
        
        # Create response prompt
        response_prompt = f"""
        Customer Question: {query}
        Query Type: {query_type.value}
        
        Relevant Knowledge Base Information:
        {' '.join(knowledge)}
        
        {data_context}
        
        Provide a helpful, comprehensive response that:
        1. Directly answers the customer's question
        2. Uses the knowledge base information
        3. Includes relevant details and procedures
        4. Offers next steps if applicable
        5. Maintains a friendly, professional tone
        
        If the question requires account-specific information, politely explain that 
        they need to contact the office for personalized assistance.
        """
        
        try:
            ai_response = self.model.generate_content(response_prompt)
            
            # Add helpful contact information for account-specific needs
            response_text = ai_response.text
            
            if query_type in [CustomerQueryType.PAYMENT_INQUIRY, CustomerQueryType.LEASE_INFO]:
                response_text += "\n\n*For account-specific information, please contact our office at (555) 123-4567 during business hours.*"
            
            return {
                'response': response_text,
                'query_type': query_type.value,
                'knowledge_used': knowledge,
                'data_used': not data.empty,
                'success': True
            }
            
        except Exception as e:
            return {
                'response': "I'd be happy to help! For specific questions, please contact our office at (555) 123-4567 during business hours (Monday-Friday 8AM-6PM, Saturday 9AM-4PM).",
                'error': str(e),
                'success': False
            }

# Streamlit UI for Simplified Customer Chatbot
def main():
    st.set_page_config(
        page_title="🏠 Tenant Information Portal", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for clean design
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Tenant Information Portal</h1>
        <p>Get instant answers about policies, procedures, and general information</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    
    # Database setup
    uploaded_file = st.sidebar.file_uploader("Upload Database (Admin Only)", type=["db", "sqlite"])
    if uploaded_file:
        db_path = "/tmp/customer_db.db"
        with open(db_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        db_path = "database.db"  # Default database
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = SimplifiedCustomerChatbot(db_path)
    
    # Quick Questions Section
    st.subheader("🚀 Quick Information")
    
    quick_questions = [
        ("💰 Rent Payment", "How do I pay my rent and when is it due?"),
        ("🔧 Maintenance", "How do I submit a maintenance request?"),
        ("📋 Lease Terms", "What are the standard lease terms and policies?"),
        ("🏊 Amenities", "What amenities are available and what are the hours?"),
        ("📞 Contact Info", "What are the office hours and contact information?"),
        ("🚪 Move In/Out", "What are the move-in and move-out procedures?"),
        ("🅿️ Parking", "What are the parking rules and guest policies?"),
        ("🐕 Pets", "What is the pet policy?")
    ]
    
    cols = st.columns(4)
    for i, (title, question) in enumerate(quick_questions):
        with cols[i % 4]:
            if st.button(title, key=f"quick_{i}", use_container_width=True):
                st.session_state.suggested_query = question
    
    # Main Chat Interface
    st.subheader("💬 Ask Any Question")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Recent Conversation")
        for user_msg, bot_response, timestamp in st.session_state.chat_history[-3:]:
            st.markdown(f"""
            <div class="user-message">
                <strong>You ({timestamp.strftime('%H:%M')}):</strong> {user_msg}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="bot-message">
                <strong>Assistant:</strong> {bot_response}
            </div>
            """, unsafe_allow_html=True)
    
    # Query input
    user_query = st.text_area(
        "What would you like to know?",
        placeholder="e.g., 'When is rent due?', 'How do I submit maintenance requests?', 'What are the pool hours?'",
        height=100,
        value=st.session_state.get('suggested_query', '')
    )
    
    if 'suggested_query' in st.session_state:
        del st.session_state.suggested_query
    
    # Submit button
    if st.button("💬 Get Answer", type="primary", use_container_width=True) and user_query:
        with st.spinner("Finding the information for you..."):
            # Process query
            result = st.session_state.chatbot.process_query(user_query)
            
            # Add to chat history
            timestamp = datetime.now()
            st.session_state.chat_history.append((
                user_query,
                result['response'],
                timestamp
            ))
            
            # Display result
            if result['success']:
                st.write("**Assistant:**")
                st.write(result['response'])
                
                # Show additional information if query type suggests it
                query_type = result.get('query_type', '')
                
                if query_type == 'maintenance_request':
                    st.error("🚨 Emergency Situations - Call immediately: (555) 123-EMERGENCY")
                    st.write("• Water leaks or flooding")
                    st.write("• No heat or air conditioning") 
                    st.write("• Electrical issues or power outages")
                    st.write("• Gas leaks")
                    st.write("• Security concerns")
                
                elif query_type == 'payment_inquiry':
                    st.info("💳 Payment Methods Available")
                    st.write("• Online portal (24/7)")
                    st.write("• Bank transfer/ACH")
                    st.write("• Credit/Debit card")
                    st.write("• Check or money order") 
                    st.write("• Automatic payment setup")
                
                elif query_type == 'contact_info':
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info("🏢 Office Information")
                        st.write("**Phone:** (555) 123-4567")
                        st.write("**Email:** info@propertymanagement.com")
                        st.write("**Emergency:** (555) 123-EMERGENCY")
                    with col2:
                        st.info("🕐 Office Hours")
                        st.write("**Mon-Fri:** 8AM - 6PM")
                        st.write("**Saturday:** 9AM - 4PM")
                        st.write("**Sunday:** Closed")
            
            else:
                st.error("❌ I had trouble processing your question. Please try rephrasing or contact our office directly.")
    
    # Common Topics Section
    st.markdown("---")
    st.subheader("📚 Common Topics")
    
    with st.expander("💰 Rent and Payments", expanded=False):
        st.markdown("""
        - **Due Date:** 1st of each month
        - **Late Fee:** $50 after the 5th
        - **Payment Methods:** Online, bank transfer, check, money order, card
        - **Online Portal:** Available 24/7
        - **Auto-Pay:** Set up to avoid late fees
        """)
    
    with st.expander("🔧 Maintenance Requests", expanded=False):
        st.markdown("""
        - **Emergency:** Call (555) 123-EMERGENCY immediately
        - **Urgent:** Submit online, 24-48 hour response
        - **Standard:** Submit online, 3-5 business days
        - **Categories:** Plumbing, Electrical, HVAC, Appliances, General
        - **Hours:** Mon-Fri 8AM-5PM, Emergency 24/7
        """)
    
    with st.expander("📋 Lease Information", expanded=False):
        st.markdown("""
        - **Standard Term:** 12 months
        - **Renewal Notice:** Offered 90 days before expiration
        - **Early Termination:** 60 days notice + fee
        - **Rent Increases:** 60 days written notice required
        - **Subletting:** Requires written management approval
        """)
    
    with st.expander("🏊 Amenities & Facilities", expanded=False):
        st.markdown("""
        - **Pool:** Seasonal hours (May-September), 6AM-10PM
        - **Fitness Center:** 6AM-10PM daily with key fob
        - **Laundry:** Coin-operated, available 24/7
        - **Community Room:** Available for resident events
        - **Parking:** One assigned space + guest parking
        """)
    
    # Footer
    st.markdown("---")
    st.info("**Need Personal Account Help?**")
    st.write("For account-specific questions, payments, or personal assistance:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("📞 **Office:** (555) 123-4567")
        st.write("🚨 **Emergency:** (555) 123-EMERGENCY")
    with col2:
        st.write("🕒 **Office Hours:** Mon-Fri 8AM-6PM, Sat 9AM-4PM")
        st.write("📧 **Email:** info@propertymanagement.com")
    
    st.caption("This portal provides general information. For specific account details, please contact our office.")

if __name__ == "__main__":
    main()
