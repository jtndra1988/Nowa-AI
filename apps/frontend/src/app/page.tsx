"use client";

import React from "react";
import MarsBotUI from "@/components/MarsBotUI";
import LoginPanel from "@/components/auth/LoginPanel";
import { useSession } from "@/lib/profile";

export default function Page() {
  const { session } = useSession();

  // Still require login
  if (!session) {
    return <LoginPanel />;
  }

  // Go straight into the main app – no onboarding wizard overlay
  return <MarsBotUI />;
}
