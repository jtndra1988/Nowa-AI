"use client";
import { motion } from "framer-motion";

export default function NeonToggle({
  enabled,
  onToggle,
}: {
  enabled: boolean;
  onToggle: (value: boolean) => void;
}) {
  return (
    <motion.div
      className="w-16 h-8 rounded-full cursor-pointer border border-white/20 relative"
      onClick={() => onToggle(!enabled)}
      animate={{
        background: enabled
          ? "linear-gradient(90deg,#22c55e,#3b82f6)"
          : "rgba(255,255,255,0.08)",
      }}
    >
      <motion.div
        className="w-7 h-7 bg-white rounded-full absolute top-0.5"
        animate={{
          x: enabled ? 36 : 2,
          boxShadow: enabled ? "0 0 15px #22c55e" : "0 0 5px #fff",
        }}
      />
    </motion.div>
  );
}
