"use client";

import { useChat } from "ai/react";
import { Chat } from "@/components/ui/chat";
import { useEffect, useState } from "react";

// Define types for our nutrition response data
type Source = {
  id: number;
  content_preview: string;
  source: string;
  type: string;
  name?: string;
};

type Citation = {
  id: number;
  text: string;
  source_name: string;
  source_url: string | null;
  date_accessed: string;
  preview: string;
};

/**
 * Chat interface component for the Wise Nutrition application
 * Connects to the backend API through the Next.js API route
 */
export function ChatDemo() {
  const [mounted, setMounted] = useState(false);
  const [activeSources, setActiveSources] = useState<Source[]>([]);
  const [activeCitations, setActiveCitations] = useState<Citation[]>([]);
  
  // Initialize chat with our API endpoint
  const { 
    messages, 
    input, 
    handleInputChange, 
    handleSubmit, 
    isLoading,
    stop,
    append
  } = useChat({
    api: "/api/chat",
    streamProtocol: 'text', // Configure to handle plain text streams
    initialMessages: [
      {
        id: "welcome-message",
        role: "assistant",
        content: "Hello! I'm your nutrition advisor. Ask me anything about healthy eating, nutrients, or dietary advice."
      }
    ],
    onFinish: (message) => {
      // Log the entire message object received when the API call finishes
      console.log('onFinish received message:', JSON.stringify(message, null, 2));

      // When a message is finished, check for sources and citations in the response data
      if (message.role === 'assistant' && message.id) {
        // Extract sources and citations from the message data if available
        const sources = (message as any).sources;
        const citations = (message as any).citations;
        
        if (sources) {
          setActiveSources(sources);
        }
        
        if (citations) {
          setActiveCitations(citations);
        }
      }
    }
  });

  // Handle hydration issues with SSR
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  // Custom handler for submitting messages
  const handleChatSubmit = (event?: { preventDefault?: () => void }) => {
    if (event?.preventDefault) {
      event.preventDefault();
    }
    // Clear sources and citations when sending a new message
    setActiveSources([]);
    setActiveCitations([]);
    handleSubmit(event);
  };

  return (
    <div className="flex flex-col w-full max-w-4xl mx-auto">
      <div className="flex-1 overflow-y-auto">
        <Chat
          messages={messages}
          input={input}
          handleInputChange={handleInputChange}
          handleSubmit={handleChatSubmit}
          isGenerating={isLoading}
          stop={stop}
          append={append}
          suggestions={[
            "What foods are high in vitamin D?",
            "Explain the benefits of a Mediterranean diet",
            "How much protein should I eat daily?",
          ]}
        />
      </div>
      
      {/* Display sources and citations if available */}
      {(activeSources.length > 0 || activeCitations.length > 0) && (
        <div className="mt-6 border-t pt-4 text-sm">
          {activeSources.length > 0 && (
            <div className="mb-4">
              <h3 className="font-semibold mb-2">Sources:</h3>
              <ul className="space-y-2">
                {activeSources.map((source) => (
                  <li key={source.id} className="p-2 bg-muted/50 rounded">
                    <div className="font-medium">{source.name || source.source}</div>
                    <div className="text-xs opacity-70">{source.content_preview}</div>
                    <div className="text-xs mt-1">Type: {source.type}</div>
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {activeCitations.length > 0 && (
            <div>
              <h3 className="font-semibold mb-2">Citations:</h3>
              <ul className="space-y-2">
                {activeCitations.map((citation) => (
                  <li key={citation.id} className="p-2 bg-muted/50 rounded">
                    <div className="text-xs">{citation.text}</div>
                    <div className="text-xs mt-1 opacity-70">{citation.preview}</div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
