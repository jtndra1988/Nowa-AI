// This line tells TypeScript that any file ending with .jsx is a valid module.
// It declares that these modules will have a default export of type 'any'.
// This is the bridge that allows your .tsx files to import .jsx files
// without needing a full type definition for the imported component.

declare module '*.jsx';
