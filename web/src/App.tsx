import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { Thread } from "@/components/assistant-ui/thread";
import { ThreadList } from "@/components/assistant-ui/thread-list";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { createLlemeTransport } from "@/lib/lleme-transport";
import { useChatWithPersistence } from "@/lib/use-chat-with-persistence";
import {
  useState,
  useMemo,
  useEffect,
  type FC,
  type ComponentPropsWithRef,
} from "react";
import { ModelSelector } from "@/components/model-selector";
import { useModels } from "@/lib/lleme-api";
import { MenuIcon, MessagesSquare, PanelLeftIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import type { TooltipContentProps } from "@radix-ui/react-tooltip";

type ButtonWithTooltipProps = ComponentPropsWithRef<typeof Button> & {
  tooltip: string;
  side?: TooltipContentProps["side"];
};

const ButtonWithTooltip: FC<ButtonWithTooltipProps> = ({
  children,
  tooltip,
  side = "top",
  ...rest
}) => {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button {...rest}>
          {children}
          <span className="sr-only">{tooltip}</span>
        </Button>
      </TooltipTrigger>
      <TooltipContent side={side}>{tooltip}</TooltipContent>
    </Tooltip>
  );
};

const Logo: FC = () => {
  return (
    <div className="flex items-center gap-2 px-2 font-medium text-sm">
      <MessagesSquare className="size-5" />
      <span className="text-foreground/90">lleme</span>
    </div>
  );
};

const Sidebar: FC<{ collapsed?: boolean }> = ({ collapsed }) => {
  return (
    <aside
      className={cn(
        "flex h-full flex-col bg-sidebar transition-all duration-200",
        collapsed ? "w-0 overflow-hidden opacity-0" : "w-[260px] opacity-100",
      )}
    >
      <div className="flex h-14 shrink-0 items-center px-4">
        <Logo />
      </div>
      <div className="flex-1 overflow-y-auto p-3">
        <ThreadList />
      </div>
    </aside>
  );
};

const MobileSidebar: FC = () => {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="size-9 shrink-0 md:hidden"
        >
          <MenuIcon className="size-4" />
          <span className="sr-only">Toggle menu</span>
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="w-[280px] p-0">
        <div className="flex h-14 items-center px-4">
          <Logo />
        </div>
        <div className="p-3">
          <ThreadList />
        </div>
      </SheetContent>
    </Sheet>
  );
};

const Header: FC<{
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
  model: string;
  onModelChange: (model: string) => void;
}> = ({ sidebarCollapsed, onToggleSidebar, model, onModelChange }) => {
  return (
    <header className="flex h-14 shrink-0 items-center gap-2 px-4">
      <MobileSidebar />
      <ButtonWithTooltip
        variant="ghost"
        size="icon"
        tooltip={sidebarCollapsed ? "Show sidebar" : "Hide sidebar"}
        side="bottom"
        onClick={onToggleSidebar}
        className="hidden size-9 md:flex"
      >
        <PanelLeftIcon className="size-4" />
      </ButtonWithTooltip>
      <ModelSelector value={model} onChange={onModelChange} />
    </header>
  );
};

export default function App() {
  const [model, setModel] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { models } = useModels();

  useEffect(() => {
    const first = models[0];
    if (!first) return;
    if (model && models.some((m) => m.id === model)) return;
    setModel(first.id);
  }, [model, models]);

  const transport = useMemo(() => createLlemeTransport({ model }), [model]);

  const runtime = useChatWithPersistence({
    transport,
    model,
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex h-dvh w-full bg-background">
        <div className="hidden md:block">
          <Sidebar collapsed={sidebarCollapsed} />
        </div>
        <div className="flex flex-1 flex-col overflow-hidden">
          <Header
            sidebarCollapsed={sidebarCollapsed}
            onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
            model={model}
            onModelChange={setModel}
          />
          <main className="flex-1 overflow-hidden">
            <Thread />
          </main>
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
}
