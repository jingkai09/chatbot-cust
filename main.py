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
    PERSONAL_ACCOUNT = "personal_account"
    OVERDUE_PAYMENT = "overdue_payment"
    MAINTENANCE_ISSUE = "maintenance_issue"

class TenantIdentifier:
    """Simple tenant identification system"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def find_tenant_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find tenant by first name, last name, or full name"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Try different name matching approaches
            queries = [
                # Full name match
                """
                SELECT t.id, t.first_name, t.last_name, t.email, t.phone,
                       l.id as lease_id, l.unit_id, l.start_date, l.end_date, 
                       l.rent_amount, l.security_deposit, l.status as lease_status,
                       u.unit_number, u.bedrooms, u.bathrooms, u.square_feet,
                       p.name as property_name, p.address_line1, p.city, p.state
                FROM tenants t
                LEFT JOIN leases l ON t.id = l.tenant_id AND l.status = 'active'
                LEFT JOIN units u ON l.unit_id = u.id
                LEFT JOIN properties p ON u.property_id = p.id
                WHERE LOWER(t.first_name || ' ' || t.last_name) LIKE LOWER(?)
                """,
                # First name match
                """
                SELECT t.id, t.first_name, t.last_name, t.email, t.phone,
                       l.id as lease_id, l.unit_id, l.start_date, l.end_date, 
                       l.rent_amount, l.security_deposit, l.status as lease_status,
                       u.unit_number, u.bedrooms, u.bathrooms, u.square_feet,
                       p.name as property_name, p.address_line1, p.city, p.state
                FROM tenants t
                LEFT JOIN leases l ON t.id = l.tenant_id AND l.status = 'active'
                LEFT JOIN units u ON l.unit_id = u.id
                LEFT JOIN properties p ON u.property_id = p.id
                WHERE LOWER(t.first_name) LIKE LOWER(?)
                """,
                # Last name match
                """
                SELECT t.id, t.first_name, t.last_name, t.email, t.phone,
                       l.id as lease_id, l.unit_id, l.start_date, l.end_date, 
                       l.rent_amount, l.security_deposit, l.status as lease_status,
                       u.unit_number, u.bedrooms, u.bathrooms, u.square_feet,
                       p.name as property_name, p.address_line1, p.city, p.state
                FROM tenants t
                LEFT JOIN leases l ON t.id = l.tenant_id AND l.status = 'active'
                LEFT JOIN units u ON l.unit_id = u.id
                LEFT JOIN properties p ON u.property_id = p.id
                WHERE LOWER(t.last_name) LIKE LOWER(?)
                """
            ]
            
            search_terms = [f"%{name}%", f"%{name}%", f"%{name}%"]
            
            for query, term in zip(queries, search_terms):
                df = pd.read_sql_query(query, conn, params=[term])
                if not df.empty:
                    conn.close()
                    return df.iloc[0].to_dict()
            
            conn.close()
            return None
            
        except Exception as e:
            print(f"Error finding tenant: {e}")
            return None

class DatabaseQueryEngine:
    """Query engine for database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_tenant_payment_info(self, tenant_id: int) -> Dict[str, Any]:
        """Get payment information for a specific tenant"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get current balance and payment info
            query = """
            SELECT 
                SUM(CASE WHEN paid_on IS NULL THEN amount ELSE 0 END) as outstanding_balance,
                SUM(CASE WHEN paid_on IS NULL AND due_date < date('now') THEN amount ELSE 0 END) as overdue_amount,
                COUNT(CASE WHEN paid_on IS NULL THEN 1 END) as unpaid_invoices,
                MIN(CASE WHEN paid_on IS NULL THEN due_date END) as next_due_date,
                MAX(CASE WHEN paid_on IS NOT NULL THEN paid_on END) as last_payment_date,
                SUM(CASE WHEN paid_on IS NOT NULL THEN amount ELSE 0 END) as total_paid
            FROM payments p
            JOIN leases l ON p.lease_id = l.id
            WHERE l.tenant_id = ?
            """
            
            df = pd.read_sql_query(query, conn, params=[tenant_id])
            conn.close()
            
            if not df.empty:
                row = df.iloc[0]
                return {
                    'outstanding_balance': row['outstanding_balance'] or 0,
                    'overdue_amount': row['overdue_amount'] or 0,
                    'unpaid_invoices': row['unpaid_invoices'] or 0,
                    'next_due_date': row['next_due_date'],
                    'last_payment_date': row['last_payment_date'],
                    'total_paid': row['total_paid'] or 0
                }
            
            return {'outstanding_balance': 0, 'overdue_amount': 0, 'unpaid_invoices': 0}
            
        except Exception as e:
            print(f"Error getting payment info: {e}")
            return {'error': str(e)}
    
    def get_tenant_maintenance_tickets(self, tenant_id: int) -> pd.DataFrame:
        """Get maintenance tickets for a specific tenant"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT st.id, st.category, st.subcategory, st.description, 
                   st.status, st.priority, st.created_at, st.updated_at
            FROM service_tickets st
            JOIN leases l ON st.lease_id = l.id
            WHERE l.tenant_id = ?
            ORDER BY st.created_at DESC
            """
            
            df = pd.read_sql_query(query, conn, params=[tenant_id])
            conn.close()
            return df
            
        except Exception as e:
            print(f"Error getting maintenance tickets: {e}")
            return pd.DataFrame()
    
    def get_lease_details(self, tenant_id: int) -> Optional[Dict[str, Any]]:
        """Get current lease details for a tenant"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT l.*, u.unit_number, u.bedrooms, u.bathrooms, u.square_feet,
                   p.name as property_name, p.address_line1, p.city, p.state
            FROM leases l
            JOIN units u ON l.unit_id = u.id
            JOIN properties p ON u.property_id = p.id
            WHERE l.tenant_id = ? AND (l.status = 'active' OR l.status IS NULL)
            ORDER BY l.created_at DESC
            LIMIT 1
            """
            
            df = pd.read_sql_query(query, conn, params=[tenant_id])
            conn.close()
            
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
            
        except Exception as e:
            print(f"Error getting lease details: {e}")
            return None

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

class DatabaseChatbot:
    """Database-connected customer chatbot"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.tenant_identifier = TenantIdentifier(db_path)
        self.db_engine = DatabaseQueryEngine(db_path)
        self.knowledge_base = CustomerKnowledgeBase()
        
        self.model = genai.GenerativeModel(
            "gemini-2.0-flash-exp",
            system_instruction=self._get_system_prompt()
        )
        
        # Track current tenant session
        self.current_tenant = None
    
    def _get_system_prompt(self) -> str:
        return """You are a helpful customer service assistant for a property management company. 
You can access real tenant data to provide specific answers about:

1. PERSONAL ACCOUNT INFO: Lease details, contract dates, unit information
2. PAYMENT STATUS: Current balance, overdue amounts, payment history  
3. MAINTENANCE REQUESTS: Current tickets, status updates

GUIDELINES:
- Be friendly, professional, and helpful
- Use actual data when available to give specific answers
- For general questions, provide policy information
- Always protect tenant privacy - only discuss the current tenant's information
- Be empathetic about payment or maintenance issues

When you have tenant data, provide specific, accurate information.
When you don't have data, guide them to contact the office."""
    
    def identify_tenant_from_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract tenant name from query and find in database"""
        
        # Look for "I am [name]" or "My name is [name]" patterns
        name_patterns = [
            r"i am (\w+(?:\s+\w+)?)",
            r"my name is (\w+(?:\s+\w+)?)", 
            r"this is (\w+(?:\s+\w+)?)",
            r"i'm (\w+(?:\s+\w+)?)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, query.lower())
            if match:
                name = match.group(1)
                tenant_info = self.tenant_identifier.find_tenant_by_name(name)
                if tenant_info:
                    return tenant_info
        
        return None
    
    def set_current_tenant(self, tenant_info: Dict[str, Any]):
        """Set the current tenant for the session"""
        self.current_tenant = tenant_info
    
    def process_query_with_context(self, query: str) -> Dict[str, Any]:
        """Process query with tenant context"""
        
        # First, try to identify tenant from query
        if not self.current_tenant:
            tenant_info = self.identify_tenant_from_query(query)
            if tenant_info:
                self.set_current_tenant(tenant_info)
                return {
                    'response': f"Hi {tenant_info['first_name']}! I found your account. How can I help you today?",
                    'tenant_identified': True,
                    'tenant_info': tenant_info,
                    'success': True
                }
        
        # Classify query type
        query_type = self.classify_query(query)
        
        # Route to appropriate handler
        if query_type == CustomerQueryType.PERSONAL_ACCOUNT:
            return self._handle_personal_account_with_data(query)
        elif query_type == CustomerQueryType.OVERDUE_PAYMENT:
            return self._handle_payment_with_data(query)
        elif query_type == CustomerQueryType.MAINTENANCE_ISSUE:
            return self._handle_maintenance_with_data(query)
        else:
            # Use knowledge base for general questions
            return self._handle_general_question_with_knowledge(query, query_type)
    
    def _handle_personal_account_with_data(self, query: str) -> Dict[str, Any]:
        """Handle personal account questions with real data"""
        
        if not self.current_tenant:
            return {
                'response': "I'd be happy to help with your account information! Please tell me your name first (e.g., 'I am John Smith') so I can look up your details.",
                'requires_identification': True,
                'success': True
            }
        
        query_lower = query.lower()
        tenant_id = self.current_tenant['id']
        
        if any(word in query_lower for word in ['expire', 'end', 'expiry', 'ends', 'contract']):
            # Get lease details
            lease_info = self.db_engine.get_lease_details(tenant_id)
            
            if lease_info:
                end_date = lease_info['end_date']
                unit_number = lease_info['unit_number']
                property_name = lease_info['property_name']
                
                response = f"""Hi {self.current_tenant['first_name']}! Here are your lease details:

**Your Lease Information:**
🏠 **Property:** {property_name}
🚪 **Unit:** {unit_number}
📅 **Lease End Date:** {end_date}
💰 **Monthly Rent:** ${lease_info['rent_amount']:.2f}
🛡️ **Security Deposit:** ${lease_info['security_deposit']:.2f}

**Important Notes:**
• Renewal offers are typically sent 90 days before expiration
• If you want to renew, contact us as early as possible
• 30-60 days notice required if you plan to move out

**Need to discuss renewal or have questions?**
📞 Call our office: (555) 123-4567
📧 Email: info@propertymanagement.com"""
                
                return {
                    'response': response,
                    'lease_data': lease_info,
                    'query_type': 'personal_account',
                    'success': True
                }
            else:
                return {
                    'response': f"Hi {self.current_tenant['first_name']}! I'm having trouble finding your current lease information. Please contact our office at (555) 123-4567 for assistance with your lease details.",
                    'success': True
                }
        
        elif any(word in query_lower for word in ['unit', 'apartment', 'room']):
            lease_info = self.db_engine.get_lease_details(tenant_id)
            
            if lease_info:
                response = f"""Hi {self.current_tenant['first_name']}! Here's your unit information:

**Your Unit Details:**
🏠 **Property:** {lease_info['property_name']}
🚪 **Unit Number:** {lease_info['unit_number']}
🛏️ **Bedrooms:** {lease_info['bedrooms']}
🛁 **Bathrooms:** {lease_info['bathrooms']}
📐 **Square Feet:** {lease_info['square_feet']} sq ft
📍 **Address:** {lease_info['address_line1']}, {lease_info['city']}, {lease_info['state']}

**Need help with your unit?**
• Maintenance requests: Submit online or call (555) 123-4567
• Unit modifications: Require written approval
• Key replacement: $25 fee, available during office hours"""
                
                return {
                    'response': response,
                    'unit_data': lease_info,
                    'success': True
                }
        
        return {
            'response': f"Hi {self.current_tenant['first_name']}! I can help with your account information. What specifically would you like to know about your lease, unit, or account?",
            'success': True
        }
    
    def _handle_payment_with_data(self, query: str) -> Dict[str, Any]:
        """Handle payment questions with real data"""
        
        if not self.current_tenant:
            return {
                'response': "I'd be happy to help with your payment information! Please tell me your name first (e.g., 'I am John Smith') so I can look up your account.",
                'requires_identification': True,
                'success': True
            }
        
        tenant_id = self.current_tenant['id']
        payment_info = self.db_engine.get_tenant_payment_info(tenant_id)
        
        if 'error' in payment_info:
            return {
                'response': f"Hi {self.current_tenant['first_name']}! I'm having trouble accessing your payment information right now. Please call our office at (555) 123-4567 for immediate assistance with your account balance.",
                'success': True
            }
        
        outstanding = payment_info['outstanding_balance']
        overdue = payment_info['overdue_amount']
        unpaid_count = payment_info['unpaid_invoices']
        next_due = payment_info['next_due_date']
        
        if outstanding > 0:
            if overdue > 0:
                response = f"""Hi {self.current_tenant['first_name']}! Here's your current payment status:

⚠️ **URGENT - You have overdue payments:**
💰 **Total Outstanding:** ${outstanding:.2f}
🚨 **Overdue Amount:** ${overdue:.2f}
📄 **Unpaid Invoices:** {unpaid_count}
📅 **Next Due Date:** {next_due if next_due else 'N/A'}

**Immediate Action Required:**
📞 **Call NOW:** (555) 123-4567 to discuss payment arrangements
💻 **Pay Online:** Use your tenant portal for immediate payment
🏢 **Visit Office:** Mon-Fri 8AM-6PM, Sat 9AM-4PM

**Important:**
• Late fees of $50 apply after the 5th of each month
• Additional late fees may accrue on overdue amounts
• Contact us immediately to avoid further penalties

**Payment Methods:**
• Online portal (fastest)
• Phone payment: (555) 123-4567
• Check/money order at office
• Bank transfer/ACH"""
            else:
                response = f"""Hi {self.current_tenant['first_name']}! Here's your current payment status:

💰 **Current Balance:** ${outstanding:.2f}
📄 **Unpaid Invoices:** {unpaid_count}
📅 **Next Due Date:** {next_due if next_due else 'N/A'}

**Good news:** No overdue amounts!

**Payment Options:**
💻 **Online Portal:** Available 24/7
📞 **Phone:** (555) 123-4567
🏢 **Office:** Drop off check or money order
🏦 **Auto-Pay:** Set up to never miss a payment

**Payment Due:** 1st of each month
**Late Fee:** $50 after the 5th"""
        else:
            response = f"""Hi {self.current_tenant['first_name']}! Great news about your account:

✅ **Account Status:** All payments current!
💰 **Outstanding Balance:** $0.00
📅 **No overdue amounts**

**Your next payment:** Due on the 1st of next month

**Keep it up!** Consider setting up auto-pay to maintain your perfect payment record:
💻 **Online Portal:** Set up automatic payments
📞 **Call Office:** (555) 123-4567 for auto-pay assistance"""
        
        return {
            'response': response,
            'payment_data': payment_info,
            'query_type': 'payment_inquiry',
            'success': True
        }
    
    def _handle_maintenance_with_data(self, query: str) -> Dict[str, Any]:
        """Handle maintenance questions with real data"""
        
        if not self.current_tenant:
            return {
                'response': "I'd be happy to help with maintenance information! Please tell me your name first (e.g., 'I am John Smith') so I can look up your tickets.",
                'requires_identification': True,
                'success': True
            }
        
        tenant_id = self.current_tenant['id']
        tickets = self.db_engine.get_tenant_maintenance_tickets(tenant_id)
        
        # Check if asking about status vs. reporting new issue
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['status', 'check', 'update', 'progress']):
            # They want to check existing tickets
            if tickets.empty:
                response = f"""Hi {self.current_tenant['first_name']}! You don't have any maintenance requests in our system currently.

**Need to submit a new request?**
💻 **Online Portal:** Submit 24/7
📞 **Call Office:** (555) 123-4567
📧 **Email:** maintenance@propertymanagement.com"""
            else:
                open_tickets = tickets[tickets['status'] != 'completed']
                completed_tickets = tickets[tickets['status'] == 'completed']
                
                response = f"""Hi {self.current_tenant['first_name']}! Here's the status of your maintenance requests:

**Open Requests ({len(open_tickets)}):**"""
                
                for _, ticket in open_tickets.head(5).iterrows():
                    created_date = ticket['created_at'][:10] if ticket['created_at'] else 'Unknown'
                    response += f"""
🔧 **Ticket #{ticket['id']}** - {ticket['category']}
   📝 Issue: {ticket['description'][:60]}...
   📊 Status: {ticket['status'].title()}
   ⚡ Priority: {ticket['priority'].title()}
   📅 Submitted: {created_date}"""
                
                if len(completed_tickets) > 0:
                    response += f"""

**Recently Completed ({len(completed_tickets)}):**"""
                    for _, ticket in completed_tickets.head(3).iterrows():
                        updated_date = ticket['updated_at'][:10] if ticket['updated_at'] else 'Unknown'
                        response += f"""
✅ **Ticket #{ticket['id']}** - {ticket['category']}: Completed {updated_date}"""
                
                response += """

**Questions about a specific ticket?**
📞 Call office with ticket number: (555) 123-4567"""
        else:
            # They're reporting a new issue - handle specific problems
            return self._handle_specific_maintenance_issue_with_context(query)
        
        return {
            'response': response,
            'tickets_data': tickets,
            'query_type': 'maintenance_status',
            'success': True
        }
    
    def _handle_specific_maintenance_issue_with_context(self, query: str) -> Dict[str, Any]:
        """Handle specific maintenance issues with tenant context"""
        
        query_lower = query.lower()
        tenant_name = self.current_tenant['first_name'] if self.current_tenant else "there"
        
        if any(word in query_lower for word in ['bulb', 'light', 'lighting']):
            response = f"""Hi {tenant_name}! I can help you with your lighting issue.

**For light bulb problems:**
💡 **Standard bulbs** - These are typically your responsibility to replace
⚡ **If it's electrical** - This is our responsibility to fix

**What to do:**
1. **Try a new bulb first** - Standard replacements are tenant responsibility
2. **Still not working?** - Submit a maintenance request immediately
3. **Multiple lights out?** - Likely electrical, submit urgent request

**Submit your request:**
💻 **Online Portal:** Mark as urgent if electrical
📞 **Call:** (555) 123-4567
📧 **Email:** maintenance@propertymanagement.com

**Emergency?** If there's electrical danger, call (555) 123-EMERGENCY!"""
        
        elif any(word in query_lower for word in ['leak', 'leaking', 'water']):
            response = f"""Hi {tenant_name}! Water leaks need immediate attention!

🚨 **TAKE ACTION NOW:**
1. **Turn off water** if possible
2. **Call emergency line:** (555) 123-EMERGENCY
3. **Document with photos** if safe
4. **Protect your belongings**

**This is an EMERGENCY - Don't wait!**
📞 **Call NOW:** (555) 123-EMERGENCY (24/7)

Water damage can affect other units and worsen quickly."""
        
        else:
            response = f"""Hi {tenant_name}! I can help you submit a maintenance request.

**To report your issue:**
📝 **Describe the problem clearly**
📷 **Take photos if helpful**
🏷️ **Choose priority level:**
• Emergency: Safety hazards, water leaks, no heat/AC
• Urgent: Appliances not working, plumbing issues  
• Standard: General repairs, cosmetic issues

**Submit your request:**
💻 **Online Portal:** Available 24/7 (fastest)
📞 **Call Office:** (555) 123-4567
📧 **Email:** maintenance@propertymanagement.com

**Emergency situations call:** (555) 123-EMERGENCY"""
        
        return {
            'response': response,
            'query_type': 'maintenance_issue',
            'success': True
        }
    
    def classify_query(self, query: str) -> CustomerQueryType:
        """Classify query type"""
        query_lower = query.lower()
        
        # Personal account questions
        personal_keywords = ['my contract', 'my lease', 'when does my', 'my rental', 'my account', 'my unit']
        if any(keyword in query_lower for keyword in personal_keywords):
            if any(word in query_lower for word in ['overdue', 'owe', 'behind', 'late']):
                return CustomerQueryType.OVERDUE_PAYMENT
            return CustomerQueryType.PERSONAL_ACCOUNT
        
        # Payment keywords
        if any(word in query_lower for word in ['payment', 'pay', 'rent', 'due', 'overdue', 'owe', 'balance']):
            return CustomerQueryType.OVERDUE_PAYMENT
        
        # Maintenance keywords
        if any(word in query_lower for word in ['bulb', 'light', 'broken', 'leak', 'maintenance', 'repair', 'fix']):
            return CustomerQueryType.MAINTENANCE_ISSUE
        
        return CustomerQueryType.GENERAL_POLICY
    
    def _handle_general_question_with_knowledge(self, query: str, query_type: CustomerQueryType) -> Dict[str, Any]:
        """Handle general questions using knowledge base"""
        
        relevant_knowledge = self.knowledge_base.search_knowledge(query)
        
        # Generate response using knowledge base
        response_prompt = f"""Customer Question: {query}

Relevant Knowledge Base Information:
{' '.join(relevant_knowledge)}

Provide a helpful, comprehensive response using the knowledge base information.
Be friendly and professional."""
        
        try:
            ai_response = self.model.generate_content(response_prompt)
            
            return {
                'response': ai_response.text,
                'query_type': query_type.value,
                'knowledge_used': relevant_knowledge,
                'success': True
            }
            
        except Exception as e:
            return {
                'response': "I'd be happy to help! For specific questions, please contact our office at (555) 123-4567 during business hours.",
                'error': str(e),
                'success': False
            }

# Streamlit UI for Customer Chatbot
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
        st.session_state.chatbot = DatabaseChatbot(db_path)
    
    # Initialize current tenant
    if 'current_tenant' not in st.session_state:
        st.session_state.current_tenant = None
    
    # Tenant identification section
    if not st.session_state.current_tenant:
        st.info("👋 **Welcome!** To get personalized help with your account, please tell me your name in the chat below (e.g., 'I am John Smith' or 'My name is Sarah Johnson')")
    
    else:
        # Show current tenant info
        tenant = st.session_state.current_tenant
        st.success(f"👋 Welcome back, **{tenant['first_name']} {tenant['last_name']}**!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"🏠 **Property:** {tenant.get('property_name', 'N/A')}")
        with col2:
            st.write(f"🚪 **Unit:** {tenant.get('unit_number', 'N/A')}")
        with col3:
            if st.button("🔄 Switch Tenant", type="secondary"):
                st.session_state.current_tenant = None
                st.session_state.chatbot.current_tenant = None
                st.rerun()
    
    # Quick Actions (updated for database functionality)
    st.subheader("🚀 Quick Actions")
    
    if st.session_state.current_tenant:
        # Personalized quick actions
        quick_questions = [
            ("📅 My Contract", "When does my contract expire?"),
            ("💰 Payment Status", "What's my current balance?"),
            ("🔧 My Maintenance", "Check my maintenance requests"),
            ("🏠 My Unit Info", "Tell me about my unit"),
            ("💸 Any Overdue?", "Do I have any overdue payments?"),
            ("📞 Contact Office", "What are the office hours?"),
            ("💡 Bulb Issue", "My bulb is damaged, what should I do?"),
            ("🚰 Water Leak", "I have a water leak in my unit")
        ]
    else:
        # General quick actions
        quick_questions = [
            ("💰 Payment Info", "How do I pay my rent?"),
            ("🔧 Maintenance", "How do I submit a maintenance request?"),
            ("📋 Lease Terms", "What are the standard lease terms?"),
            ("📞 Contact Info", "What are the office hours?"),
            ("🏊 Amenities", "What amenities are available?"),
            ("🅿️ Parking", "What are the parking rules?"),
            ("🚪 Moving", "What's the move-out process?"),
            ("🐕 Pets", "What's the pet policy?")
        ]
    
    cols = st.columns(4)
    for i, (title, question) in enumerate(quick_questions):
        with cols[i % 4]:
            if st.button(title, key=f"quick_{i}", use_container_width=True):
                st.session_state.suggested_query = question
    
    # Main Chat Interface
    st.subheader("💬 Ask Any Question")
    
    # Display chat history in a selectbox dropdown
    if st.session_state.chat_history:
        st.markdown("### 📜 Conversation History")
        
        # Create options for the dropdown
        history_options = ["Select a conversation to view..."]
        for i, (user_msg, bot_response, timestamp) in enumerate(reversed(st.session_state.chat_history)):
            # Truncate long messages for dropdown display
            short_msg = user_msg[:50] + "..." if len(user_msg) > 50 else user_msg
            option = f"{timestamp.strftime('%H:%M')} - {short_msg}"
            history_options.append(option)
        
        # Dropdown selection
        selected_conversation = st.selectbox(
            "Previous conversations:",
            options=history_options,
            key="conversation_selector"
        )
        
        # Display selected conversation
        if selected_conversation != "Select a conversation to view...":
            selected_index = history_options.index(selected_conversation) - 1
            if 0 <= selected_index < len(st.session_state.chat_history):
                # Get the conversation (accounting for reversed order)
                conv_index = len(st.session_state.chat_history) - 1 - selected_index
                user_msg, bot_response, timestamp = st.session_state.chat_history[conv_index]
                
                # Display the selected conversation
                st.markdown("**Selected Conversation:**")
                st.info(f"🧑‍💻 **You ({timestamp.strftime('%H:%M')}):** {user_msg}")
                st.success(f"🤖 **Assistant:** {bot_response}")
                st.divider()
    
    # Query input
    user_query = st.text_area(
        "What would you like to know?",
        placeholder="e.g., 'I am John Smith', 'When does my contract expire?', 'What's my current balance?'",
        height=100,
        value=st.session_state.get('suggested_query', '')
    )
    
    if 'suggested_query' in st.session_state:
        del st.session_state.suggested_query
    
    # Submit button
    if st.button("💬 Get Answer", type="primary", use_container_width=True) and user_query:
        with st.spinner("Finding the information for you..."):
            # Set current tenant in chatbot if we have one
            if st.session_state.current_tenant:
                st.session_state.chatbot.current_tenant = st.session_state.current_tenant
            
            # Process query
            result = st.session_state.chatbot.process_query_with_context(user_query)
            
            # Handle tenant identification
            if result.get('tenant_identified'):
                st.session_state.current_tenant = result['tenant_info']
                st.session_state.chatbot.current_tenant = result['tenant_info']
                st.success("✅ Tenant identified! Ask me anything about your account.")
                st.rerun()
            
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
                
                # Show additional data if available
                if 'lease_data' in result:
                    st.subheader("📄 Your Lease Details")
                    lease = result['lease_data']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Monthly Rent", f"${lease['rent_amount']:.2f}")
                        st.metric("Security Deposit", f"${lease['security_deposit']:.2f}")
                    with col2:
                        st.metric("Bedrooms", lease['bedrooms'])
                        st.metric("Square Feet", f"{lease['square_feet']} sq ft")
                
                if 'payment_data' in result:
                    payment = result['payment_data']
                    st.subheader("💰 Payment Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Outstanding Balance", f"${payment['outstanding_balance']:.2f}")
                    
                    with col2:
                        st.metric("Overdue Amount", f"${payment['overdue_amount']:.2f}")
                    
                    with col3:
                        st.metric("Unpaid Invoices", payment['unpaid_invoices'])
                    
                    if payment['overdue_amount'] > 0:
                        st.error("⚠️ **Action Required:** You have overdue payments. Please contact the office immediately!")
                
                if 'tickets_data' in result and not result['tickets_data'].empty:
                    st.subheader("🔧 Your Maintenance Requests")
                    tickets = result['tickets_data']
                    
                    # Display in a nice table format
                    display_tickets = tickets[['id', 'category', 'description', 'status', 'priority', 'created_at']].copy()
                    display_tickets['created_at'] = pd.to_datetime(display_tickets['created_at']).dt.strftime('%Y-%m-%d')
                    display_tickets.columns = ['Ticket #', 'Category', 'Description', 'Status', 'Priority', 'Created']
                    
                    st.dataframe(display_tickets, use_container_width=True)
                
                # Show unit data if available
                if 'unit_data' in result:
                    st.subheader("🏠 Your Unit Information")
                    unit = result['unit_data']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Unit:** {unit['unit_number']}")
                        st.write(f"**Property:** {unit['property_name']}")
                    with col2:
                        st.write(f"**Bedrooms:** {unit['bedrooms']}")
                        st.write(f"**Bathrooms:** {unit['bathrooms']}")
                    with col3:
                        st.write(f"**Square Feet:** {unit['square_feet']}")
                        st.write(f"**Address:** {unit['address_line1']}")
            
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
