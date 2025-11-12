// src/app/page.tsx
"use client";

import React from "react";
import MarsBotUI from "@/components/MarsBotUI";
import LoginPanel from "@/components/auth/LoginPanel";
import OnboardingWizard from "@/components/onboarding/OnboardingWizard";
import { useSession, useClientProfile } from "@/lib/profile";

export default function Page() {
  const { session } = useSession();
  const { profile } = useClientProfile();

  if (!session) {
    return <LoginPanel />;
  }

  return (
    <>
      <MarsBotUI />
      {(!profile || !profile.isOnboarded) && <OnboardingWizard />}
    </>
  );
}
