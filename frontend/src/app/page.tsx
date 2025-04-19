import { ChatDemo } from "@/components/chat-demo";

/**
 * Main page component for the Wise Nutrition application
 */
export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center w-full h-[calc(100vh-2rem)] px-4 py-2">
      <div className="flex flex-col space-y-4 max-w-4xl w-full h-full">
        <h1 className="text-2xl font-bold tracking-tight">Wise Nutrition Advisor</h1>
        <p className="text-muted-foreground">Ask questions about nutrition, healthy eating, vitamins, minerals, and more.</p>
        
        <div className="flex-1 overflow-hidden rounded-md border">
          <ChatDemo />
        </div>
      </div>
    </div>
  );
}
