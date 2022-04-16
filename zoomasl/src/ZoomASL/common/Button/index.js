import React from "react";
import { ButtonContainer } from "./styles";

function Button({text,onClick,variant}) {
  return <ButtonContainer variant={variant} onClick={onClick}>
    {text}
  </ButtonContainer>;
}
export default Button;