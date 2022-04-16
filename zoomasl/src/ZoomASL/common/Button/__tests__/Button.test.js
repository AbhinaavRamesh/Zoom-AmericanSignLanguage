import Button from "../index";
import { shallow } from "enzyme";
import React from 'react';
describe("Button", () => {
  it("should render correctly", () => {
    let ButtonSnapshot = shallow(<Button/>);
    expect(ButtonSnapshot).toMatchSnapshot();
  });
});