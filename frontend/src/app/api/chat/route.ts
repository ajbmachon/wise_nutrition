import { NextRequest, NextResponse } from 'next/server';

/**
 * Handles chat API requests and proxies them to the backend
 * 
 * @param req The incoming Next.js request
 * @returns Response with the chat response (streaming)
 */
export async function POST(req: NextRequest) {
  console.log('API route handler started');
  try {
    const body = await req.json();
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const sessionId = body.id || `session-${Date.now()}`;
    

    const latestMessage = body.messages[body.messages.length - 1].content;
    
    console.log('Sending request to backend:', `${apiUrl}/api/v1/nutrition_rag_chain/invoke`);
    console.log('Query:', latestMessage);
    console.log('Session ID:', sessionId);

    // Format the input for LangServe - direct format without nesting
    const input = {
      query: latestMessage,
      session_id: sessionId
    };

    // Call our backend API
    const response = await fetch(`${apiUrl}/api/v1/nutrition_rag_chain/invoke`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Backend API error (${response.status}):`, errorText);
      
      return NextResponse.json({
        id: sessionId,
        role: 'assistant',
        content: `I'm having trouble connecting to the nutrition database. Please try again later. (Error: ${response.status})`,
        createdAt: new Date(),
      });
    }

    // Read the full JSON response from the backend
    const data = await response.json();
    console.log('Backend response:', data);

    // Get the response text from the backend data
    const responseText = data.response || 'Received response, but no content found.';
    console.log('Creating plain text stream with:', responseText.substring(0, 100) + '...');
    
    // Store metadata in memory for later retrieval
    // Since we can't include it in the headers with plain text streaming
    if (data.sources && data.sources.length > 0) {
      console.log('Response includes sources:', data.sources.length);
      // In a production app, you would store this in a database or cache
    }
    
    if (data.citations && data.citations.length > 0) {
      console.log('Response includes citations:', data.citations.length);
      // In a production app, you would store this in a database or cache
    }
    
    // Create a simple text encoder for the stream
    const encoder = new TextEncoder();
    
    // Create a ReadableStream that outputs the response text
    const stream = new ReadableStream({
      async start(controller) {
        console.log('Stream controller started');
        
        // Send the text in small chunks to simulate streaming
        const chunkSize = 3; // Characters per chunk
        for (let i = 0; i < responseText.length; i += chunkSize) {
          const chunk = responseText.slice(i, i + chunkSize);
          controller.enqueue(encoder.encode(chunk));
          
          // Small delay between chunks
          await new Promise(resolve => setTimeout(resolve, 15));
          
          // Log progress occasionally
          if (i % 100 === 0 && i > 0) {
            console.log(`Streamed ${i}/${responseText.length} characters`);
          }
        }
        
        console.log('Stream completed, closing controller');
        controller.close();
      },
    });

    // Return a simple text/plain response with the stream
    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
      },
    });

  } catch (error) {
    console.error('Error in chat API route:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    
    return NextResponse.json({
      id: `error-${Date.now()}`,
      role: 'assistant',
      content: `I encountered an error processing your request. Please try again. (${errorMessage})`,
      createdAt: new Date(),
    }, { status: 200 }); // Return 200 to client but with error message
  }
}
