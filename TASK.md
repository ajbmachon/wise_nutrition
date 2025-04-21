# Wise Nutrition: Task Management

This document tracks the current tasks, backlog items, and milestones for the Wise Nutrition project. It will be updated as work progresses.

## Current Active Tasks

### Project Setup
- [x] Create AI agent with Pydantic AI (2025-04-21)
- [ ] Initialize git repository and set up GitHub project
- [ ] Create conda/virtual environment with base dependencies
- [ ] Setup Docker Compose for local development
- [ ] Configure linting and formatting tools
- [ ] Set up CI workflow for automated testing

### Infrastructure
- [ ] Create Supabase project
- [ ] Configure pgvector extension
- [ ] Set up authentication system
- [ ] Implement row-level security policies
- [ ] Configure vector indexes for optimal performance

### Data Preparation
- [x] Create document processing pipeline (2025-04-21)
- [x] Implement chunking strategy for retrieval (2025-04-21)
- [ ] Collect and organize nutrition information sources
- [ ] Build embedding generation system
- [ ] Develop data ingestion script

### RAG System Development
- [x] Define agent architecture and components (2025-04-21)
- [x] Create query understanding and parsing module (2025-04-21)
- [x] Implement simple vector search functionality (2025-04-21)
- [x] Build context assembly system (2025-04-21)
- [x] Develop response generation with citations (2025-04-21)
- [x] Create evaluation metrics and testing framework (2025-04-21)

### Backend API
- [ ] Set up FastAPI project structure
- [ ] Implement Pydantic AI orchestration patterns
- [ ] Build chat endpoint and logic
- [ ] Create recipe generation endpoint
- [ ] Develop user profile management
- [ ] Add questionnaire processing endpoint

### Frontend Development
- [ ] Initialize Next.js project with shadcn/ui
- [ ] Design and implement chat interface
- [ ] Create recipe display and generation UI
- [ ] Build optional questionnaire interface
- [ ] Implement authentication flows
- [ ] Add user profile management screens

## Backlog Items

### Enhanced Features
- [ ] Implement conversational follow-up questions
- [ ] Add semantic search filters (e.g., by nutrient, condition)
- [ ] Create specialized dietary pattern modules (e.g., paleo, keto within Weston Price framework)
- [ ] Develop meal planning functionality
- [ ] Add shopping list generation
- [ ] Implement seasonal food awareness
- [ ] Build cooking technique education module

### Content Expansion
- [ ] Expand nutrition knowledge base
- [ ] Add more recipe templates
- [ ] Include traditional food preparation techniques
- [ ] Create nutrient profiles for common foods
- [ ] Develop content on food quality and sourcing

### Technical Improvements
- [ ] Implement caching layer for common queries
- [ ] Optimize embedding generation for performance
- [ ] Add multi-model support (use different LLMs for different tasks)
- [ ] Implement token usage tracking and optimization
- [ ] Create admin dashboard for system monitoring

### User Experience
- [ ] Add image generation for recipes
- [ ] Implement progressive disclosure of complex topics
- [ ] Create onboarding flow for new users
- [ ] Develop mobile-optimized interface
- [ ] Add dark mode support

## Milestones

### Phase 1: MVP RAG System
- Project setup completed
- Basic infrastructure configured
- Initial knowledge base ingested
- Simple RAG query-response working
- Basic UI for chat interaction

### Phase 2: Recipe Generation
- Recipe generation algorithm implemented
- Recipe templates integrated
- Nutrition knowledge linked to recipe suggestions
- UI for recipe display and customization

### Phase 3: User Personalization
- User accounts and authentication
- Questionnaire system for nutrition assessment
- Personalized recommendations based on profile
- History tracking and preference learning

### Phase 4: Enhanced Features & Optimization
- Improved retrieval with hybrid search
- Expanded knowledge base and recipe collection
- Performance optimizations
- Additional specialized features

## Notes & Discoveries
- Consider chunking strategy: nutrition information likely needs semantic chunking rather than just size-based
- May need custom embedding model or fine-tuning for nutrition-specific language
- Hybrid search implementation will be critical for balancing relevance and accuracy
- Need to research best practices for citation generation in RAG systems

## Discovered During Work
- [ ] Implement the production-ready vector store with Supabase pgvector
- [ ] Create a more sophisticated search algorithm with hybrid search
- [ ] Add nutritional profile for users

## Completed Tasks
- [x] Create AI agent with Pydantic AI for nutrition information and recipe recommendations (2025-04-21)
- [x] Implement basic retrieval mechanism for demonstration (2025-04-21)
- [x] Create unit tests for retrieval and agent components (2025-04-21)
- [x] Build interactive CLI for querying the nutrition agent (2025-04-21)

## Usage Instructions

### Running the Application

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY=your-api-key
```

3. Run the interactive mode:
```bash
python main.py interactive
```

4. Or query with a specific question:
```bash
python main.py query "What are good sources of vitamin D?"
```

### Running Tests

```bash
pytest tests/
```
