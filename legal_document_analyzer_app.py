import streamlit as st
import PyPDF2
from agno.agent import Agent
from agno.models.google import Gemini

# Configure API key
API_KEY = "AIzaSyDmcPbEDAEojTomYs7vLKu107fOa7c6500"

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Mortality Review Agent
mortality_reviewer = Agent(
    name="Mortality Review Agent",
    model=Gemini(id="gemini-2.0-flash-exp", api_key=API_KEY),
    markdown=True,
    description="Expert in analyzing clinical cases for quality improvement, with focus on identifying deviations from standard of care, preventable factors in mortality/morbidity, and system-level improvements.",
    instructions=[
        "Analyze the given clinical case thoroughly using this systematic approach:",
        
        "1. CASE TIMELINE RECONSTRUCTION",
        "- Create detailed chronological timeline of events",
        "- Highlight key clinical decision points",
        "- Document timing of interventions",
        "- Note delays or gaps in care",
        
        "2. STANDARD OF CARE ANALYSIS",
        "- Compare actions against current clinical guidelines",
        "- Identify deviations from standard protocols",
        "- Document timing metrics vs. established benchmarks",
        "- Flag missed diagnostic opportunities",
        
        "3. CONTRIBUTING FACTORS ANALYSIS",
        "- System issues (staffing, resources, communication)",
        "- Clinical decision-making gaps",
        "- Documentation/handoff problems",
        "- Equipment/technical factors",
        "- Team coordination issues",
        
        "4. PREVENTABILITY ASSESSMENT",
        "- Classify events as preventable, potentially preventable, or non-preventable",
        "- Identify key intervention points that could have changed outcome",
        "- Assess impact of each deviation on final outcome",
        "- Document uncertainty in preventability assessment",
        
        "Format your response using markdown with these sections:",
        "# Mortality Review Analysis",
        "## Case Summary",
        "## Timeline of Events",
        "## Standard of Care Deviations",
        "## Contributing Factors",
        "## Preventability Analysis",
        "## System Improvement Recommendations",
        "## Learning Points",
        
        "For each identified deviation:",
        "- Link to specific guidelines/standards",
        "- Document potential impact on outcome",
        "- Suggest specific preventive measures",
        "- Note system-level implications",
        
        "Flag events that require:",
        "- Immediate system-level review",
        "- Policy changes",
        "- Educational interventions",
        "- Equipment/resource modifications",
        "- Communication system improvements",
    ],
)

# Update page configuration
st.set_page_config(layout="wide", page_title="Mortality Review Analysis", page_icon="🏥")
st.title("🔍 Clinical Case Mortality Review")

# Create columns for input options
col1, col2 = st.columns(2)

with col1:
    st.info("Upload clinical case document (PDF)")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Add specific analysis options
    analysis_focus = st.multiselect(
        "Select areas for detailed analysis:",
        ["Timeline Analysis", "Standard of Care Review", "System Factors", 
         "Preventability Assessment", "Quality Improvement Opportunities"]
    )
    
    # Add severity classification
    severity_level = st.selectbox(
        "Case Severity Classification:",
        ["Death", "Permanent Harm", "Temporary Harm", "Near Miss"]
    )

with col2:
    # Add specific query options
    query_type = st.radio(
        "Analysis Focus:",
        ["Comprehensive Review", "Specific Standard of Care Deviation", 
         "System Issues", "Preventability Analysis"]
    )
    
    additional_notes = st.text_area(
        "Additional Notes or Specific Concerns:",
        height=100,
        help="Enter any specific aspects of the case you want to focus on"
    )

if uploaded_file is not None and analysis_focus:
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    if st.button("Analyze Case"):
        with st.spinner("Conducting mortality review analysis..."):
            # Prepare the analysis request
            analysis_request = f"""
            Clinical Case Analysis Request:
            Severity Level: {severity_level}
            Analysis Focus: {query_type}
            Specific Areas: {', '.join(analysis_focus)}
            Additional Notes: {additional_notes}

            Case Content:
            {pdf_text}
            """
            
            response = mortality_reviewer.run(analysis_request)
            
            if response and hasattr(response, 'content'):
                st.markdown(response.content)
            else:
                st.error("Failed to generate analysis. Please try again.")

st.markdown("---")
st.markdown("Powered by Gemini and Agno | Clinical Case Review Tool")
